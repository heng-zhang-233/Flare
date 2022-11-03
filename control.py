from gym_uav import *
import time


# implementation of control algorithms


def naive_control_uav(env, seed=None):
    observation = env.reset(seed=seed)
    begin = time.time()

    time_total = 4
    fps = 50
    C = 2.0
    total_reward = 0.0

    angle = []
    total_torque = []
    angle_velocity = []

    last_angle = observation['orientation']
    print(f'total physical time to execute: {time_total}')

    for step in range(time_total * fps):
        action = {"edge_index": 2,
                  "direction": math.pi / 4, "thrust": 5.0}
        # Note: auxiliary info is acquired through `info``
        observation, reward, done, info = env.step(action)

        angle.append(observation['orientation'])
        total_torque.append(env.uav.getTotalTorque())
        angle_velocity.append(
            math.fabs(observation['orientation'] - last_angle) * fps)
        last_angle = observation['orientation']

        print(f'[{step}] getTotalTorque(): {info["totalTorque"]}')

        total_reward += reward

        still_open = env.render()
        if still_open is False or done:
            break

        if step % 1 == 0:
            print(f'[{step}] action     : {action}')
            print(f'[{step}] observation: {observation}')
            print(f'[{step}] time       : {step / fps}')

            print(f'[{step}] angle vel calculated outside: {angle_velocity[-1]}')
        if step % 3 == 0 and step > 0:
            print(f'[{step}] simulation environment is reset')
            observation = env.reset(seed=seed)

    print("--- Simulation Finished ---")
    print(f'total reward: {total_reward}')
    print(f'execution time: {time.time() - begin}')
    env.close()


if __name__ == '__main__':
    naive_control_uav(UAVDemo(seed=42))
