#!/usr/bin/env python
import shutil
import os
import math
import pdb
from tkinter import OFF
import gym
from gym import spaces
import pygame
from pygame import gfxdraw
import numpy as np
from typing import Optional
import time
import MRAC_ILC as ilc
import scipy as sp
from scipy import linalg

SIM_FPS = 50
RENDER_FPS = 10
SCALE = 100.0

HEIGHT = 1280
WIDTH = 720
UAV_SIZE = 1  # m

MAX_ANG_V = math.pi * 900  # pi/4 / sec


class UAVPhyWrapper:
    def __init__(self) -> None:
        # The distance between thrust and center(minimum)
        # i.e. (side length)/2
        self.length = 0.1

        # Phy parameters
        self.angle = 0  # rad(scalar)
        self.angular_speed = 0  # rad/s(scalar)
        self.position = np.zeros(2, dtype=np.float64)  # m(vec)
        self.speed = np.zeros(2, dtype=np.float64)  # m/s(vec)
        self.m = 1.0  # kg(scalar)
        self.inertia = 2 / 3 * self.m * self.length ** 2
        self.fixed = True
        self.last_direction = 0

    def _torque(self, action):
        d_cs = 0.0149564*4
        # print(f'Print action in _torque here: {action}')
        # The edge index is neglected
        # V_e = action.get("thrust", 10)
        V_e = action[1]
        # direction w.r.t. the edge normal
        # direction = action.get("direction", 0)
        direction = np.clip(action[0], self.last_direction - MAX_ANG_V / SIM_FPS,
                            self.last_direction + MAX_ANG_V / SIM_FPS)
        # d_cs * V_e**2 * \int_{-l}^{l}{\sqrt{l^2 + x^2} * \sin(\alpha - \theta) \, \mathrm{d} x}
        return -d_cs * self.length * V_e ** 2 * np.sin(direction)

    def _force(self, action):
        d_cs = 0.0149564*4
        # V_e = action.get("thrust", 10)
        V_e = action[1]
        # direction = action.get("direction", 0) + self.angle
        # direction = action[0] + self.angle
        direction = np.clip(action[0], self.last_direction - MAX_ANG_V / SIM_FPS,
                            self.last_direction + MAX_ANG_V / SIM_FPS) + self.angle
        direction_vector = np.array([np.cos(direction), np.sin(direction)])
        return d_cs * 2 * self.length * V_e ** 2 * -direction_vector

    def step(self, action, dt=1 / SIM_FPS):
        # Angle
        angular_acceleration = 1 / self.inertia * (self._torque(action))
        self.angular_speed += angular_acceleration * dt

        self.angle += self.angular_speed * dt
        # pdb.set_trace()

        # Position
        # No offset
        if self.fixed:
            acceleration = np.array([0.0, 0.0])
        else:
            acceleration = 1 / self.m * (self._force(action))
        self.speed += acceleration * dt
        self.position += self.speed * dt

        # Update the actual angle
        self.last_direction = np.clip(action[0], self.last_direction - MAX_ANG_V / SIM_FPS,
                                      self.last_direction + MAX_ANG_V / SIM_FPS)

        # return {
        #     "angle": self.angle,
        #     "angular_speed": self.angular_speed,
        #     "position": self.position,
        #     "speed": self.speed
        # }

        # return np.array([self.angle, self.angular_speed, self.position[0], self.position[1], self.speed[0], self.speed[1]])
        return np.array([self.angle, self.angular_speed])

class UAVEmpirical(gym.Env):
    """
    The empirical UAV environment
    """
    metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}

    def __init__(self):
        self.uav = UAVPhyWrapper()
        self._uav_size = UAV_SIZE * SCALE
        self._height = HEIGHT
        self._width = WIDTH
        self.clock = pygame.time.Clock()
        # self.last_action = {"angle": 0, "angular_speed": 0,
        #                     "position": np.zeros(3), "speed": np.zeros(3)}

        # gym related initialization
        # (angle, angular speed, position, speed)
        # scalar_bound = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        # vector_bound = spaces.Box(
        #     low=-np.array([np.inf, np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float64)
        # self.observation_space = spaces.Dict({
        #     "angle": scalar_bound,
        #     "angular_speed": scalar_bound,
        #     "position": vector_bound,
        #     "speed": vector_bound
        # })

        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)

        # self.action_space = spaces.Dict({
        #     "edge_index": spaces.Discrete(1),
        #     "direction": spaces.Box(low=-math.pi / 2, high=math.pi / 2, shape=(1,), dtype=np.float64),
        #     "thrust": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)
        # })

        self.action_space = spaces.Box(low=np.array([-math.pi / 4, 5]), high=np.array([math.pi / 4, 10]),
                                       dtype=np.float32)

        self.center = np.array([self._height // 2, self._width // 2])
        self.last_action = np.array([0, 0])

        # Pygame related initialization
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption('Empirical UAV Environment')
        self.screen = pygame.display.set_mode((self._height, self._width))
        self.running = True

        # moving average
        self.position_queue = [self.uav.position.copy()]

    def reset(self, *, seed: Optional[int] = None):
        # self.uav.angle = 0  # rad(scalar)
        # rand_reset = np.random.random()
        self.uav.angle = 0  # np.pi/3
        self.uav.angular_speed = 0  # rad/s(scalar)
        self.uav.position = np.zeros(2, dtype=np.float64)  # m(vec)
        self.uav.speed = np.zeros(2, dtype=np.float64)  # m/s(vec)
        self.last_action = np.array([0, 0])
        self.position_queue = [self.uav.position.copy()]

        return self.uav.step(self.last_action)

    def step(self, action):
        self.last_action = action
        # print(np.shape(self.uav.angle), np.shape(self.uav.angular_speed))
        observation = self.uav.step(action)

        # print(np.shape(self.uav.angle))
        reward = -math.fabs(self.uav.angle)
        if math.fabs(self.uav.angle) > math.pi / 2:
            done = True
            reward += -3
        else:
            done = False
        # return observation, reward, done, info
        return (observation, reward, done, {})

    def render(self, mode="human"):
        # Handle the events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return self.running

        # pygame.transform.rotate()
        self.surf = pygame.Surface(self.screen.get_size())
        # Set the background
        pygame.draw.rect(self.surf, color=(15, 17, 26),
                         rect=self.surf.get_rect())

        if len(self.position_queue) > 50:
            self.position_queue = self.position_queue[1:]
        self.position_queue.append(self.uav.position.copy())
        # Draw the rectangle
        OFFSET = sum(self.position_queue[:5]) / 5 * -SCALE
        CENTER = self.center + self.uav.position * SCALE + OFFSET
        WIDTH = self._uav_size
        ROTATE = self.uav.angle + np.pi / 4
        l_ = WIDTH / math.sqrt(2)
        rect = np.array([
            [np.cos(ROTATE), np.sin(ROTATE)],
            [-np.sin(ROTATE), np.cos(ROTATE)],
            [-np.cos(ROTATE), -np.sin(ROTATE)],
            [np.sin(ROTATE), -np.cos(ROTATE)]]) * l_
        # Broadcast
        # print(f'----------{np.shape(CENTER)}')
        rect += np.array(CENTER)
        pygame.gfxdraw.aapolygon(
            self.surf, rect, (255, 255, 255))
        thrust_vec_center = (rect[0] + rect[3]) / 2
        thrust_vec_direction = self.uav.angle + self.last_action[0]
        thrust_vec_end = thrust_vec_center + \
                         np.array([np.cos(thrust_vec_direction),
                                   np.sin(thrust_vec_direction)]) * 50
        pygame.draw.aaline(self.surf, (143, 218, 141),
                           thrust_vec_center, thrust_vec_end)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        # > the function will delay to keep the game running
        # > slower than the given ticks per second
        self.clock.tick(self.metadata["render_fps"])

        # Flip the self.screen to screen
        pygame.display.flip()

        return self.running

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.running = False


def main():
    shutil.rmtree('result')
    os.mkdir('result')

    env = UAVEmpirical()
    # check_env(env)

    # print(env.action_space.sample())

    steps = 0
    epoch = 0
    rate = SIM_FPS
    run_time = 20
    iteration_number = 10

    A_ref = np.matrix('0 1; -25 -10')
    B_ref = np.matrix('0; 1')  # b = d_cs * l / (2/3 * m * l^2)
    x_ref = np.matrix('0; 0')
    ys = 0

    # discrete plant dynamics (computed using continuous system)
    Ts = 1 / rate
    n = run_time * rate
    A_ref_d = sp.linalg.expm(A_ref * Ts)
    B_ref_d = np.linalg.pinv(A_ref) * (A_ref_d - np.eye(np.shape(A_ref)[0])) * B_ref

    while epoch < iteration_number:
        print(f'New epoch[{epoch}] start now!')
        env.reset(seed=42)
        x_ref = np.matrix('0; 0')
        ilc_ilc = ilc.MRAC_ILC(rate, run_time)
        P = sp.linalg.solve_discrete_lyapunov(A_ref_d, ilc_ilc.Q * Ts)
        # print(ilc_ilc.Kx_hat)


        while steps < run_time * rate:
            t = steps * 1 / SIM_FPS

            ym = x_ref
            # d_ym = x_ref.T[0, 1]
            ilc_ilc.ILC_law(ym, steps)

            ys = ilc_ilc.input
            # yr = ilc_ilc.result['angle'][-1]
            x_ref = A_ref_d * x_ref + B_ref_d * ys
            # print(ym)
            #ilc_ilc.result['ym'].append(x_ref.T[0, 1])

            a = np.array([-ilc_ilc.alpha, ilc_ilc.Ve])
            s, r, done, info = env.step(a)
            ilc_ilc.control_law(s, x_ref, steps)
            # pdb.set_trace()

            xp = s.reshape((-1,1))

            # ilc_ilc.w1 = (- ilc_ilc.w1 + ilc_ilc.input_MRAC) * Ts + ilc_ilc.w1
            # ilc_ilc.w2 = (- ilc_ilc.w2 + ilc_ilc.result['orientation'][-1]) * Ts + ilc_ilc.w2
            # w = np.array([ilc_ilc.w1, ilc_ilc.w2, ilc_ilc.result['orientation'][-1], ys]).reshape(-1,1)
            # ilc_ilc.phi = (- ilc_ilc.p * ilc_ilc.phi + w) * Ts + ilc_ilc.phi
            # ilc_ilc.theta = (- ilc_ilc.Gamma * (ilc_ilc.result['orientation'][-1] - x_ref.T[0,0]) * ilc_ilc.phi) * Ts + ilc_ilc.theta
            ilc_ilc.K_x_hat = -ilc_ilc.G_x * xp * np.transpose(xp - x_ref) * P * B_ref_d * Ts + ilc_ilc.K_x_hat
            ilc_ilc.K_r_hat = -ilc_ilc.G_r * ys * np.transpose(xp - x_ref) * P * B_ref_d * Ts + ilc_ilc.K_r_hat
            # ilc_ilc.P = np.matrix('1 0; 0 1')
            # ilc_ilc.Kx_hat = ilc_ilc.Kx_hat
            # ilc_ilc.Kr_hat = ilc_ilc.Kr_hat


            if steps % (SIM_FPS // RENDER_FPS) == 0:
                still_open = env.render()
                if still_open is False:
                    break

            steps += 1
        ilc_ilc.save_result()
        epoch += 1
        steps = 0
        print(ilc_ilc.theta)
        time.sleep(1)
    ilc.plot_result(ilc_ilc.tra_flag)


if __name__ == '__main__':
    main()
