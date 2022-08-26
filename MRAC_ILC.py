import os
import pdb
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

tra = ['sin_', 'step_', 'step1_']

def plot_result(tra_flag):
    file = os.listdir("./result")
    cont = 0
    RMS = []
    RMS2 = []
    for j in range(len(file)):
        cont += 1
        for i in file:
            file_name = 'result_' + tra[tra_flag] +str(cont) + '.pkl'
            if re.match(file_name, i) is not None:
                file_result = load_obj(i)
                RMS.append(sum(file_result['error']))
                RMS2.append(sum(file_result['error_MRAC']))
                plt.figure(1)
                plt.plot(file_result['time'], file_result['error'])
                plt.xlabel('time [s]')
                plt.ylabel('error^2 [rad^2]')
                plt.legend(['Itera=' + str(i) for i in range(len(file))])
                plt.figure(2)
                plt.plot(file_result['time'], file_result['input'])
                plt.xlabel('time [s]')
                plt.ylabel('input[rad]')
                plt.legend(['Itera=' + str(i) for i in range(len(file))])
                plt.figure(3)
                file_result['orientation'].pop(0)
                file_result['orientation'].pop(1)
                plt.plot(file_result['time'], file_result['orientation'], label = 'Itera=' + str(cont - 1))
                plt.xlabel('time [s] ')
                plt.ylabel('angle [rad]')
                # plt.legend(['Itera=' + str(i) for i in range(len(file))])
                # plt.figure(5)
                # plt.plot(file_result['time'], file_result['d_orientation'], label = 'Itera' + str(cont - 1))
                # plt.xlabel('time [s] ')
                # plt.ylabel('velocity [rad/s]')
                # plt.legend(['Itera=' + str(i) for i in range(len(file))])
                plt.figure(6)
                plt.plot(file_result['time'], file_result['ym'], ls = '-.', label='ym_Itera=' + str(cont - 1))
                plt.xlabel('time [s] ')
                plt.ylabel('ym [rad]')
                plt.figure(7)
                plt.plot(file_result['time'], file_result['error_MRAC'])
                plt.xlabel('time [s]')
                plt.ylabel('(yp_ym)^2 [rad^2]')
                plt.legend(['Itera=' + str(i) for i in range(len(file))])
    #         torque = [file_result['total_torque'][i] - file_result['Specific_Edge_Torque'][i] for i in range(200)]
    # io.savemat('./result/distur.mat', {'disturbance': np.mat(torque)})
    # plt.figure(10)
    # plt.plot(file_result['time'], torque)
    plt.figure(4)
    plt.plot([i for i in range(cont-1)], RMS)
    plt.xlabel('iteration time ')
    plt.ylabel('RMS_r_yp')
    plt.figure(8)
    plt.plot([i for i in range(cont - 1)], RMS2)
    plt.xlabel('iteration time ')
    plt.ylabel('RMS_ym_yp')
    plt.figure(3) # plot the desired trajectory
    file_result['angle'].pop(0)
    file_result['angle'].pop(1)
    plt.plot(file_result['time'], file_result['angle'], ls = '--', label = 'Desired_tra')
    plt.xlabel('time [s] ')
    plt.ylabel('angle [rad]')
    plt.legend()
    # plt.figure(5)
    # file_result['velocity'].pop(0)
    # file_result['velocity'].pop(1)
    # plt.plot(file_result['time'], file_result['velocity'], ls = '--', label = 'Desired_tra')
    # plt.xlabel('time [s] ')
    # plt.ylabel('velocity [rad/s]')
    # plt.legend()
    plt.figure(6)  # plot the desired trajectory
    # file_result['angle'].pop(0)
    # file_result['angle'].pop(1)
    plt.plot(file_result['time'], file_result['angle'], ls='--', label='Desired_tra')
    plt.xlabel('time [s] ')
    plt.ylabel('angle [rad]')
    plt.legend()

    plt.show()


def Change_last_data(tra_flag, itera_time):
    file = os.listdir("./result")
    tra_file = []
    for i in file:
        file_name = 'result_' + tra[tra_flag] + '*'
        if re.match(file_name, i) is not None:
            tra_file.append(i)
    result = load_obj(tra_file[itera_time])
    save_obj(result['input'], tra[tra_flag] + 'last_input')


def save_obj(obj, name):
    f = open('./result/' + name + '.pkl', 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load_obj(name):
    f = open('./result/' + name, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def load_last(num, flag):
    file = os.listdir("./result")
    for i in file:
        file_name = tra[flag] + 'last_input*'
        if re.match(file_name, i) is None:
            last_list_input = [0 for k in range(num)]
        else:
            last_list_input = load_obj(i)
            return last_list_input
    last_list_input = [0 for k in range(num)]
    return last_list_input


class MRAC_ILC:
    # MRAC_ILC controller
    def __init__(self, rate, run_time):
        self.rate = rate
        self.delay = 50
        self.acc = 2
        self.KP = 10
        self.KD = 10

        self.phi = np.matrix('0; 0; 0; 0')
        self.theta = 1 * np.matrix('1; 1; 1; 1')
        self.w1 = 0
        self.w2 = 0
        self.Gamma = 500 * np.identity(4)
        self.p = 2

        self.K_x_hat = np.matrix('0;0')
        self.K_r_hat = np.matrix('0')
        self.Phi_hat = np.matrix('0')
        self.Q = np.matrix('10 0; 0 10')
        self.G_x = 1 * np.matrix('1 0; 0 1')
        self.G_r = 1 * np.matrix('1')
        self.G_p = np.matrix('1')

        self.tra_flag = 0
        self.max_value = 0
        self.step_input = [np.pi / 3]
        self.input = 0
        self.input_MRAC = 0
        self.Ve = 5
        self.alpha = 0
        self.last_input = 0
        self.last_list_input = load_last(rate * run_time, self.tra_flag)
        self.result = {'error': [], 'error_ym': [], 'error_MRAC': [], 'input': [], 'input_MRAC': [], 'time': [], 'angle': [0 for i in range(2)],
                       'velocity': [0 for i in range(2)], 'orientation': [0, 0],
                       'd_orientation': [], 'total_torque': [], 'Specific_Edge_Torque': [],
                       'ym': [], 'd_ym': []}

    def control_law(self, xp, x_ref, step):
        ys = self.input
        # ys = self.result['angle'][-1]
        orientation = xp[0]
        d_orientation = xp[1]
        # d_orientation = (3 * orientation - 4 * self.result['orientation'][step + 1] + self.result['orientation'][step]) \
        #                 * self.rate * 0.5
        self.result['orientation'].append(orientation)
        self.result['d_orientation'].append(d_orientation)
        # u = self.KP * (self.result['ym'][-1] - orientation) + self.KD * (self.result['d_ym'][-1] - d_orientation)
        # xp = np.array([self.result['orientation'][-1], self.result['d_orientation'][-1]]).reshape((-1,1))
        xp = xp.reshape((-1,1))
        u = (np.transpose(self.K_x_hat) * xp + self.K_r_hat * ys)[0,0]
        u = u - self.KP * orientation - self.KD * d_orientation
        # w = np.array([self.w1, self.w2, orientation, ys]).reshape(-1, 1)
        # d_theta = - self.Gamma * (orientation - x_ref.T[0,0]) * self.phi
        # u = (np.transpose(self.theta) * w + np.transpose(d_theta) * self.phi)[0, 0]

        # pdb.set_trace()
        self.result['input_MRAC'].append(u)
        # if np.abs(u) >= 0.5:
        #     u = 0.5 * np.sign(u)
        self.Ve = np.sqrt(2 * (2 + np.abs(u)))
        self.alpha = np.arcsin(u / (2 * (2 + np.abs(u))))

        self.input_MRAC = u
        # self.alpha = u

        self.result['error'].append((self.result['angle'][-1] - orientation) ** 2)
        self.result['error_MRAC'].append((self.result['ym'][-1] - orientation) ** 2)
        self.result['time'].append(step / self.rate)

    def ILC_law(self, x_ref, step):
        desired = self.desired_tra(step)
        # try:
        ym = x_ref.T[0,0]
        d_ym = x_ref.T[0,1]
        # d_ym = (3 * ym - 4 * self.result['ym'][step + 1] + self.result['ym'][step]) * self.rate * 0.5
        # except RuntimeWarning:
        #     pdb.set_trace()
        self.result['ym'].append(ym)
        self.result['d_ym'].append(d_ym)
        u = self.last_list_input[step] + self.KP * (desired['trajectory'] - ym) + self.KD * (
                desired['velocity'] - d_ym)
        self.result['input'].append(u)

        self.input = u
        # self.alpha = u
        self.last_input = self.input

        self.result['error_ym'].append((desired['trajectory'] - ym) ** 2)
        # self.result['time'].append(step / self.rate)

    def desired_tra(self, step):
        t = step / self.rate
        desired = {'trajectory': 0, 'velocity': 0}
        if self.tra_flag == 0:
            w = np.sqrt(self.acc)
            desired['trajectory'] = 0.2 * np.sin(w * t)
            desired['velocity'] = 0.2 * w * np.cos(w * t)

        elif self.tra_flag == 1:
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

        elif self.tra_flag == 2:
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
        cont = 0
        for i in file_name:
            if re.match('result*', i) is not None:
                cont += 1
        save_obj(self.result, 'result_' + tra[self.tra_flag] + str(cont + 1))
        save_obj(self.result['input'], tra[self.tra_flag] + 'last_input')
