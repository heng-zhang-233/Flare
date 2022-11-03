import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import io


def plot_result(tra_flag):
    tra = ['sin_', 'step_', 'step1_']
    file = os.listdir("./result")
    cont = 0
    RMS = []
    flag = 0
    for j in range(len(file)):
        cont += 1
        for i in file:
            file_name = 'result_' + tra[tra_flag] + str(cont) + '.pkl'
            if re.match(file_name, i) is not None:
                file_result = load_obj(i)
                if flag == 0:
                    plt.figure(3)
                    file_result['angle'].pop(0)
                    file_result['angle'].pop(1)
                    plt.plot(file_result['time'], file_result['angle'], ls='-')
                    plt.figure(5)
                    file_result['velocity'].pop(0)
                    file_result['velocity'].pop(1)
                    plt.plot(file_result['time'], file_result['velocity'], ls='-')
                    flag = 1

                RMS.append(np.sqrt(sum(file_result['error']) / len(file_result['error'])))
                plt.figure(1)
                plt.plot(file_result['time'], file_result['error'])
                plt.xlabel('time [s]')
                plt.ylabel('error^2 [rad^2]')
                plt.legend(['Itera=' + str(i) for i in range(len(file))])
                plt.figure(2)
                plt.plot(file_result['time'], file_result['input'])
                plt.xlabel('time [s]')
                plt.ylabel('input [rad]')
                plt.legend(['Itera=' + str(i) for i in range(len(file))])
                plt.figure(3)
                file_result['orientation'].pop(0)
                file_result['orientation'].pop(1)
                plt.plot(file_result['time'], file_result['orientation'])
                plt.xlabel('time [s] ')
                plt.ylabel('angle [rad]')
                plt.legend(['desired'] + ['Itera=' + str(i) for i in range(len(file) - 1)])
                plt.figure(5)
                plt.plot(file_result['time'], file_result['d_orientation'])
                plt.xlabel('time [s] ')
                plt.ylabel('velocity [rad/s]')
                plt.legend(['desired'] + ['Itera=' + str(i) for i in range(len(file) - 1)])
    #         torque = [file_result['total_torque'][i] - file_result['Specific_Edge_Torque'][i] for i in range(200)]
    # io.savemat('./result/distur.mat', {'disturbance': np.mat(torque)})
    # plt.figure(10)
    # plt.plot(file_result['time'], torque)
    plt.figure(4)
    plt.plot([i for i in range(cont - 1)], RMS)
    plt.xlabel('iteration time ')
    plt.ylabel('RMS')
    plt.figure(6)
    plt.plot(file_result['time'], file_result['hat_phi'])
    plt.xlabel('time [s] ')
    plt.legend(['hat_phi_1', 'hat_phi_2', 'hat_phi_3', 'hat_phi_4'])
    plt.figure(7)
    plt.plot(file_result['time'], file_result['hat_d'])
    plt.xlabel('time [s] ')
    plt.legend(['hat_d'])
    plt.show()


def Change_last_data(tra_flag, itera_time):
    tra = ["sin_", "step_", "step1_"]
    file = os.listdir("./result")
    tra_file = []
    for i in file:
        file_name = 'result_' + tra[tra_flag] + '*'
        if re.match(file_name, i) is not None:
            tra_file.append(i)
    result = load_obj(tra_file[itera_time])
    save_obj(result['input'], tra[tra_flag] + 'last_input')


def save_obj(obj, name):
    with open('./result/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load_obj(name):
    with open('./result/' + name, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


def load_last(num, flag):
    tra = ["sin_", "step_", "step1_"]
    file = os.listdir("./result")
    result = {'hat_phi': [[0, 0, 0] for k in range(num)], 'hat_d': [0 for k in range(num)],
              'input': [0 for k in range(num)]}
    for i in file:
        file_name = tra[flag - 1] + 'last_input*'
        if re.match(file_name, i) is None:
            last_list_input = result
        else:
            last_list_input = load_obj(i)
            return last_list_input
    last_list_input = result
    return last_list_input


class BLF_ILC:
    # BLF_ILC controller
    def __init__(self, rate, run_time):
        self.rate = rate
        # desired trajectary parameter
        self.delay = 50
        self.acc = 10
        # projection parameter
        self.u_start = 7
        self.phi_start = 5
        self.d_start = 2
        self.hat_phi = [0, 0, 0]
        # control law parameter
        self.KP = 4
        self.Gamma = 10
        self.gamma = 0.2
        self.lamda = 4
        self.B = 0.8974
        # trajectory flag
        self.tra_flag = 2
        if self.tra_flag == 2:
            self.max_value = 0
            self.step_input = [np.pi / 3, np.pi / 3]
        else:
            self.max_value = np.pi / 3
            self.step_input = [0, 0]

        # control input
        self.Ve = 5
        self.alpha = 0
        self.last_input = 0
        self.last_list_input = load_last(rate * run_time, self.tra_flag)
        # result
        self.result = {'error': [], 'input': [], 'time': [], 'angle': [np.pi / 3 - self.max_value for i in range(2)],
                       'velocity': [0 for i in range(2)], 'acc': [0 for i in range(2)],
                       'orientation': [np.pi / 3 - self.max_value for i in range(2)], 'd_orientation': [],
                       'total_torque': [],
                       'Specific_Edge_Torque': [], 'hat_phi': [], 'hat_d': []}

    def control_law(self, orientation, step):
        # desired trajectory
        desired = self.desired_tra(step)
        # orientation = s[0]
        # d_orientation = s[1]
        d_orientation = (3 * orientation - 4 * self.result['orientation'][step + 1] +
                         self.result['orientation'][step]) * self.rate * 0.5
        # error
        t = step / self.rate
        error = desired['trajectory'] - orientation
        d_error = desired['velocity'] - d_orientation
        # epsilon = (0.4 - 0.1) * np.exp(-0.4 * t) + 0.1
        epsilon = 0.2
        # epsilon = 0.2
        dv_de = self.KP * (1 / np.cos(np.pi * error ** 2 / (2 * epsilon ** 2)) ** 2) * error
        xi = [orientation, orientation ** 2, orientation ** 3]
        # projection
        last_input = self.project(self.last_list_input['input'][step], self.u_start)
        last_phi = self.last_list_input['hat_phi'].pop()
        last_d = self.project(self.last_list_input['hat_d'][step], self.d_start)
        # adaptive law
        if step == 0:
            self.hat_phi = last_phi
        else:
            for i in range(3):
                if self.hat_phi[i] > self.phi_start:
                    self.hat_phi[i] = self.hat_phi[i]
                else:
                    self.hat_phi[i] += self.gamma * error * xi[i] / self.rate
        hat_d = last_d + self.lamda * dv_de
        # control law
        u = dv_de + self.Gamma * d_error + self.B / (self.B ** 2) * (
                np.dot(np.mat(self.hat_phi), np.mat(xi).T)[0, 0] + desired['acc']) + hat_d
        # u = last_input + dv_de + self.Gamma * d_error
        # u=Ve^2*sin(alpha)
        self.Ve = np.sqrt(2 * (2 + np.abs(u)))
        self.alpha = np.arcsin(u / (2 * (2 + np.abs(u))))
        self.last_input = u
        # save data
        self.result['input'].append(u)
        self.result['hat_phi'].append(self.hat_phi)
        self.result['hat_d'].append(hat_d)
        self.result['error'].append((desired['trajectory'] - orientation) ** 2)
        self.result['time'].append(step / self.rate)
        self.result['orientation'].append(orientation)
        self.result['d_orientation'].append(d_orientation)

    def project(self, number, desired):
        if np.abs(number) >= desired:
            result = np.sign(number) * desired
        else:
            result = number
        return result

    def desired_tra(self, step):
        t = step / self.rate
        desired = {'trajectory': 0, 'velocity': 0, 'acc': 0}
        if self.tra_flag == 1:
            w = np.sqrt(self.acc)
            desired['trajectory'] = self.max_value * np.sin(w * t)
            desired['velocity'] = self.max_value * w * np.cos(w * t)
            desired['velocity'] = self.max_value * w * np.cos(w * t)
        elif self.tra_flag == 2:
            if step <= 2:
                desired['trajectory'] = np.pi / 3
                desired['velocity'] = 0
                desired['acc'] = 0
            else:
                desired['trajectory'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['angle'][step + 1] -
                                         self.result['angle'][step] + self.acc / (self.rate ** 2) * self.max_value) / \
                                        (1 + 2 * np.sqrt(self.acc) / self.rate + self.acc / (self.rate ** 2))
                desired['velocity'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['velocity'][step + 1] -
                                       self.result['velocity'][step] + self.acc * self.max_value / self.rate -
                                       self.acc * self.step_input[step - 2] / self.rate) / (1 + 2 * np.sqrt(self.acc) /
                                                                                            self.rate + self.acc / (
                                                                                                    self.rate ** 2))
                desired['acc'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['acc'][step + 1] -
                                  self.result['acc'][step] + self.acc * self.max_value - 2 * self.acc *
                                  self.step_input[step - 2] + self.acc * self.step_input[step - 3]) \
                                 / (1 + 2 * np.sqrt(self.acc) / self.rate + self.acc / (self.rate ** 2))
                self.step_input.append(self.max_value)
            self.result['acc'].append(desired['acc'])
        elif self.tra_flag == 3:
            if step <= 2:
                desired['trajectory'] = 0
                desired['velocity'] = 0
                desired['acc'] = 0
            else:
                desired['trajectory'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['angle'][step + 1] -
                                         self.result['angle'][step] + self.acc / (self.rate ** 2) * self.max_value) / \
                                        (1 + 2 * np.sqrt(self.acc) / self.rate + self.acc / (self.rate ** 2))
                desired['velocity'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['velocity'][step + 1] -
                                       self.result['velocity'][step] + self.acc * self.max_value / self.rate -
                                       self.acc * self.step_input[step - 2] / self.rate) / (1 + 2 * np.sqrt(self.acc) /
                                                                                            self.rate + self.acc / (
                                                                                                    self.rate ** 2))
                desired['acc'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['acc'][step + 1] -
                                  self.result['acc'][step] + self.acc * self.max_value - 2 * self.acc *
                                  self.step_input[step - 2] + self.acc * self.step_input[step - 3]) \
                                 / (1 + 2 * np.sqrt(self.acc) / self.rate + self.acc / (self.rate ** 2))
                self.step_input.append(self.max_value)
            self.result['acc'].append(desired['acc'])

        self.result['angle'].append(desired['trajectory'])
        self.result['velocity'].append(desired['velocity'])
        return desired

    def save_result(self, ):
        file_name = os.listdir("./result")
        tra = ["sin_", "step_", "step1_"]
        cont = 0
        for i in file_name:
            if re.match('result*', i) is not None:
                cont += 1
        result = {'hat_phi': self.result['hat_phi'], 'hat_d': self.result['hat_d'], 'input': self.result['input']}
        save_obj(self.result, 'result_' + tra[self.tra_flag - 1] + str(cont + 1))
        save_obj(result, tra[self.tra_flag - 1] + 'last_input')
