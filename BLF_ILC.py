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
    for j in range(len(file)):
        cont += 1
        for i in file:
            file_name = 'result_' + tra[tra_flag] +str(cont) + '.pkl'
            if re.match(file_name, i) is not None:
                file_result = load_obj(i)
                RMS.append(sum(file_result['error']))
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
                plt.legend(['Itera=' + str(i) for i in range(len(file))])
                plt.figure(5)
                plt.plot(file_result['time'], file_result['d_orientation'])
                plt.xlabel('time [s] ')
                plt.ylabel('velocity [rad/s]')
                plt.legend(['Itera=' + str(i) for i in range(len(file))])
    #         torque = [file_result['total_torque'][i] - file_result['Specific_Edge_Torque'][i] for i in range(200)]
    # io.savemat('./result/distur.mat', {'disturbance': np.mat(torque)})
    # plt.figure(10)
    # plt.plot(file_result['time'], torque)
    plt.figure(4)
    plt.plot([i for i in range(cont-1)], RMS)
    plt.xlabel('iteration time ')
    plt.ylabel('RMS')
    plt.figure(3)
    file_result['angle'].pop(0)
    file_result['angle'].pop(1)
    plt.plot(file_result['time'], file_result['angle'])
    plt.xlabel('time [s] ')
    plt.ylabel('angle [rad]')
    plt.figure(5)
    file_result['velocity'].pop(0)
    file_result['velocity'].pop(1)
    plt.plot(file_result['time'], file_result['velocity'])
    plt.xlabel('time [s] ')
    plt.ylabel('velocity [rad/s]')
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
    for i in file:
        file_name = tra[flag - 1] + 'last_input*'
        if re.match(file_name, i) is None:
            last_list_input = [0 for k in range(num)]
        else:
            last_list_input = load_obj(i)
            return last_list_input
    last_list_input = [0 for k in range(num)]
    return last_list_input

class BLF_ILC:
    # BLF_ILC controller
    def __init__(self, rate, run_time):
        self.rate = rate
        self.delay = 50
        self.acc = 20
        self.KP = 20
        self.u_start = 7
        self.gamma = 10
        self.epsilon = 0.2
        self.tra_flag = 2
        if self.tra_flag == 2:
            self.max_value = 0
        else:
            self.max_value = np.pi / 3
        self.step_input = [np.pi / 3]
        self.Ve = 5
        self.alpha = 0
        self.input = 0
        self.last_input = 0
        self.last_list_input = load_last(rate * run_time, self.tra_flag)
        self.result = {'error': [], 'input': [], 'time': [], 'angle': [np.pi / 3 for i in range(2)],
                       'velocity': [0 for i in range(2)], 'orientation': [np.pi / 3, np.pi / 3],
                       'd_orientation': [], 'total_torque': [], 'Specific_Edge_Torque': []}

    def control_law(self, orientation, step):
        desired = self.desired_tra(step)
        d_orientation = (3 * orientation - 4 * self.result['orientation'][step + 1] + self.result['orientation'][step]) \
                        * self.rate * 0.5

        error = desired['trajectory'] - orientation
        d_error = desired['velocity'] - d_orientation
        self.result['orientation'].append(orientation)
        self.result['d_orientation'].append(d_orientation)
        if np.abs(self.last_list_input[step]) >= self.u_start:
            loadinput = np.sign(self.last_list_input[step]) * self.u_start
        else:
            loadinput = self.last_list_input[step]

        u = loadinput + self.KP * (1 / np.cos(np.pi * error ** 2 / (2 * self.epsilon ** 2)) ** 2) * error + self.gamma * d_error
        self.result['input'].append(u)
        self.Ve = np.sqrt(2 * (2 + np.abs(u)))
        self.alpha = np.arcsin(u / (2 * (2 + np.abs(u))))
        # self.input = (self.delay * u / self.rate + self.last_input) / (self.delay / self.rate + 1)
        self.input = u
        self.last_input = self.input

        self.result['error'].append((desired['trajectory'] - orientation) ** 2)
        self.result['time'].append(step / self.rate)

    def desired_tra(self, step):
        t = step / self.rate
        desired = {'trajectory': 0, 'velocity': 0}
        if self.tra_flag == 1:
            w = np.sqrt(self.acc)
            desired['trajectory'] = self.max_value * np.sin(w * t)
            desired['velocity'] = self.max_value * w * np.cos(w * t)

        elif self.tra_flag == 2:
            if step <= 2:
                desired['trajectory'] = np.pi / 3
                desired['velocity'] = 0
            else:
                desired['trajectory'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['angle'][step + 1] -
                                         self.result['angle'][step] + self.acc / (self.rate ** 2) * self.max_value) / \
                                        (1 + 2 * np.sqrt(self.acc) / self.rate + self.acc / (self.rate ** 2))
                desired['velocity'] = ((2 + 2 * np.sqrt(self.acc) / self.rate) * self.result['velocity'][step + 1] -
                                       self.result['velocity'][step] + self.acc * self.max_value / self.rate -
                                       self.acc * self.step_input[step - 3] / self.rate) / (1 + 2 * np.sqrt(self.acc) /
                                                                                            self.rate + self.acc / (
                                                                                                    self.rate ** 2))
                self.step_input.append(self.max_value)

        elif self.tra_flag == 3:
            if t <= 0.1:
                desired['trajectory'] = np.pi / 3
                desired['velocity'] = 0
            else:
                desired['trajectory'] = 0
                desired['velocity'] = 0

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
        save_obj(self.result, 'result_' + tra[self.tra_flag - 1] + str(cont + 1))
        save_obj(self.result['input'], tra[self.tra_flag - 1] + 'last_input')
