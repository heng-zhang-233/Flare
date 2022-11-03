from gym_uav import *
import time
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import os
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

multiGridBool = False
multiGridParam = 3

res_init = 0.001
epoch_train_timesteps = 3e4
str_date = '0907'


def main():
    print(".py file is accessible.")

    env = UAVDemo()
    # env.set_res(res_init)
    # env.reset(seed=42)
    # check_env(env)
    print("The Gym env Test has passed.")

    cur_steps = 0
    # Here's training part.#
    ########################

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/uav-v1/")

    start = time.time()
    time_list = []
    time_list.append(start)

    if multiGridBool == False:
        env.set_res(0.004)
        env.reset(seed=42)
        total_train_timesteps = 1e5
        print(f"This time's total train timesteps is {total_train_timesteps}")
        model.learn(total_timesteps=total_train_timesteps, log_interval=4, tb_log_name='SAC_NoMG_'+str_date)
        model.save("sac_uav"+str_date)
        end = time.time()
        print("************FEEDBACK IN CONCLUSION*****************")
        print(f"This time's total train time is {end - start}")
        print(f"This time's average train time for each step is {(end - start) / total_train_timesteps} seconds.")
    else:
        print(f"Begin to do Multi-grid learning with basic_timesteps equal to {epoch_train_timesteps}")
        for res_index in range(multiGridParam):
            this_epoch_train_timesteps = int(epoch_train_timesteps / (2 ** res_index))
            this_resolution = res_init * (2 ** (multiGridParam - res_index - 1))

            env.set_res(this_resolution)
            env.reset(seed=42)

            if res_index == 0:
                model.learn(total_timesteps=this_epoch_train_timesteps, log_interval=4,
                            tb_log_name='SAC_mg_' + str_date + '_' + str(res_index))
            else:
                model.learn(total_timesteps=this_epoch_train_timesteps, log_interval=4,
                            tb_log_name='SAC_mg_' + str_date + '_' + str(res_index), reset_num_timesteps=False)

            model.save('SAC_mg_' + str_date + '_' + str(res_index))

            end = time.time()
            print(f"this iteration's total time is {end - time_list[-1]}")
            print(f"this iteration's avg time for each ts is {(end - time_list[-1]) / this_epoch_train_timesteps}")
            time_list.append(end)
            cur_steps += this_epoch_train_timesteps

        model.save('SAC_mg_' + str_date + '_' + "final")
        print("************FEEDBACK IN CONCLUSION*****************")
        print(f"For all Multi-grid learning, totally {cur_steps} are runned.")
        print(f"For all Multi-grid learning, totally {time_list[-1] - time_list[0]} seconds are consumed.")
        for res_index in range(multiGridParam):
            print(
                f"[Multi-grid's {res_index}'s learning], there're {int(epoch_train_timesteps / (2 ** res_index))} steps ran.")
            print(
                f"[Multi-grid's {res_index}'s learning], it consumed {time_list[res_index + 1] - time_list[res_index]} seconds.")
            print(
                f"[Multi-grid's {res_index}'s learning], it consumed {(time_list[res_index + 1] - time_list[res_index]) / (epoch_train_timesteps / (2 ** res_index))} seconds on each step.")

    print("Trainning and saving done. Begin to simulate then.")

    # Here's simulation part.#
    ##########################

    # model = SAC.load("sac_uav_mg_0817")
    # print("SAC trained model loaded.")
    #
    # obs = env.reset(seed=42)
    # steps = 0
    # this_iter_steps = 0
    #
    # fp_o = open("D:\\Project_EDF\\UAV_proj\\result\\obs_all.pkl","wb")
    # fp_a = open("D:\\Project_EDF\\UAV_proj\\result\\act_all.pkl","wb")
    # fp_r = open("D:\\Project_EDF\\UAV_proj\\result\\rew_all.pkl","wb")
    # fp_i = open("D:\\Project_EDF\\UAV_proj\\result\\int_all.pkl","wb")
    #
    # obs_list = []
    # act_list = []
    # rew_list = []
    # int_list = []
    #
    # int_list.append(0)
    #
    # num_simu = 1000
    # for i in range(num_simu):
    #     action, _states = model.predict(obs, deterministic=True)
    #     last_obs = obs
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     env.uav.SaveResult("D:\\Project_EDF\\UAV_proj\\result\\")
    #
    #     obs_list.append(obs)
    #     act_list.append(action)
    #     rew_list.append(reward)
    #
    #     if steps % 1 == 0 or done:
    #         print(f"[steps       ]: {steps}")
    #         print(f"[observations]: {obs}")
    #         print(f"[action      ]: {action}")
    #         print(f"[reward      ]: {reward}")
    #         # print(f"[total torque]: {env.uav.GetTotalTorque()}")
    #
    #     if done:
    #         print("Reset Done.")
    #         this_iter_steps = 0
    #         int_list.append(i)
    #         env.reset(seed=42)
    #
    #     steps += 1
    #     this_iter_steps +=1
    #
    # if (int_list[-1] != num_simu -1):
    #     int_list.append(num_simu-1)
    #
    # pickle.dump(obs_list,fp_o)
    # pickle.dump(act_list,fp_a)
    # pickle.dump(rew_list,fp_r)
    # pickle.dump(int_list,fp_i)
    #
    # fp_o.close()
    # fp_a.close()
    # fp_r.close()
    # fp_i.close()


if __name__ == '__main__':
    main()
