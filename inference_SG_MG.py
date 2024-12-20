import mujoco
import numpy as np
import random
import os
import torch
import time
from multiprocessing import Pool, Manager

from mpc_mujoco.model import MPC
from mpc_mujoco.collecting_test_model import Cartesian_Collecting_MPC
from mpc_mujoco.joint_model import Joint_MPC

import matplotlib.pyplot as plt

# Setting
CONTROL_STEPS = 200 # 200
HOR = 128
NUM_SEED = 42 # 42
MAX_CORE_CPU = 28 # 28
SAMPLING_TIME = 0.001
TARGET_POS = np.array([0.3, 0.3, 0.5])

DIFF_INI_STATES = 20 
NUM_INI_STATES = 2*DIFF_INI_STATES
# SG_INI_STATES = 10
# MG_INI_STATES = 10
MG_U_GUESS_NUMBER = 10

# data saving
RESULT_SAVED_PATH = '/root/diffusion_mujoco_panda/inference_SG_MG/test'
# COST_SG_FILENAME_SAVE = 'COST_SG.npy'
# TIME_SG_FILENAME_SAVE = 'TIME_SG.npy'
# COST_MG_FILENAME_SAVE = 'COST_MG.npy'
# TIME_MG_FILENAME_SAVE = 'TIME_MG.npy'

def main():
    # path
    # j_SG_filepath = os.path.join(RESULT_SAVED_PATH,COST_SG_FILENAME_SAVE)
    # time_SG_filepath = os.path.join(RESULT_SAVED_PATH,TIME_SG_FILENAME_SAVE)
    # j_MG_filepath = os.path.join(RESULT_SAVED_PATH,COST_MG_FILENAME_SAVE)
    # time_MG_filepath = os.path.join(RESULT_SAVED_PATH,TIME_MG_FILENAME_SAVE)

    # memories
    j_SG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200
    time_SG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200
    j_MG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200
    time_MG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200


    # initial states generating
    ini_states, ini_data_idx = ini_states_idx_generating()

    # initial data groups
    initial_data_groups = [] # SG 10 + MG 10
    for n in range(NUM_INI_STATES):
          initial_data_groups.append([ini_states[n,:], ini_data_idx[n], j_SG_memory, time_SG_memory, j_MG_memory, time_MG_memory])
    
    # pool
    with Pool(processes=MAX_CORE_CPU) as pool:
          pool.starmap(single_mpc_process, initial_data_groups)



#######################################################################################################

def ini_states_idx_generating():
    np.random.seed(NUM_SEED)
    random.seed(NUM_SEED)

    # Gaussian noise for initial states generating
    mean_ini = 0
    std_dev_ini = 0.05
    
    # initial states generating
    random_ini_states_list = []
    for i in range(DIFF_INI_STATES):
            gaussian_noise_ini_joint_1 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_2 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_3 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_4 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_5 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_6 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            random_ini_states_list.append([gaussian_noise_ini_joint_1,gaussian_noise_ini_joint_2,gaussian_noise_ini_joint_3,gaussian_noise_ini_joint_4,gaussian_noise_ini_joint_5,gaussian_noise_ini_joint_6, 0])
    random_ini_states_SG_MG_list = random_ini_states_list+ random_ini_states_list      
    random_ini_states = np.array(random_ini_states_SG_MG_list) # 20*7 

    # # random initial u guess
    # random_u_guess_list = []
    # for i in range(NUM_INI_STATES):
    #       u_guess_4 = np.round(random.uniform(-2, 2),2)
    #       u_guess_5 = np.round(random.uniform(-2, 2),2)
    #       u_guess_7 = np.round(random.uniform(-2, 2),2)
    #       random_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])
    # random_u_guess_SG_MG_list = random_u_guess_list + random_u_guess_list
    # random_ini_u_guess = np.array(random_u_guess_list) # 20*7

    # data idx
    idx_list = []
    for i in range(NUM_INI_STATES):
          idx = i
          idx_list.append(idx)
    data_idx = np.array(idx_list)

    return random_ini_states, data_idx

def SG_ini_u_guess_generating():
      # random initial u guess
      random_u_guess_list = []

      u_guess_4 = np.round(random.uniform(-2, 2),2)
      u_guess_5 = np.round(random.uniform(-2, 2),2)
      u_guess_7 = np.round(random.uniform(-2, 2),2)
      random_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])

      random_SG_ini_u_guess = np.array(random_u_guess_list) # 1*7
      return random_SG_ini_u_guess

def MG_ini_u_guess_generating():
      # random initial u guess
      random_u_guess_list = []
      for i in range(MG_U_GUESS_NUMBER):
            u_guess_4 = np.round(random.uniform(-2, 2),2)
            u_guess_5 = np.round(random.uniform(-2, 2),2)
            u_guess_7 = np.round(random.uniform(-2, 2),2)
            random_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])

      random_MG_ini_u_guess = np.array(random_u_guess_list) # 5*7
      return random_MG_ini_u_guess


def single_mpc_process(initial_state, initial_idx, j_SG_memory, time_SG_memory, j_MG_memory, time_MG_memory):
      try:
            # panda mujoco
            panda = mujoco.MjModel.from_xml_path('/root/diffusion_mujoco_panda/xml/mjx_scene.xml')
            data = mujoco.MjData(panda)
            mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

            # Simulation
            if initial_idx.item() <= 0.5*NUM_INI_STATES-1: # <=
                random_SG_ini_u_guess = SG_ini_u_guess_generating()
                joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x_collecting_ini, u_collecting_ini, delta_t = mpc.SG_simulate(random_SG_ini_u_guess,initial_state,initial_idx)

                j_SG_memory = np.array(mpc_cost).reshape(1,CONTROL_STEPS)
                time_SG_memory = np.array(delta_t).reshape(1,CONTROL_STEPS)

                j_SG_path = os.path.join(RESULT_SAVED_PATH, f'j_SG_' + 'idx-' + str(initial_idx.item()) + '_test.npy')
                np.save(j_SG_path, j_SG_memory)

                time_SG_path = os.path.join(RESULT_SAVED_PATH, f'time_SG_' + 'idx-' + str(initial_idx.item()) + '_test.npy')
                np.save(time_SG_path, time_SG_memory)
            else:
                random_MG_ini_u_guess = MG_ini_u_guess_generating()
                joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x_collecting_ini, u_collecting_ini, delta_t = mpc.MG_simulate(random_MG_ini_u_guess,initial_state,initial_idx)

                j_MG_memory = np.array(mpc_cost).reshape(1,CONTROL_STEPS)
                time_MG_memory = np.array(delta_t).reshape(1,CONTROL_STEPS)

                j_MG_path = os.path.join(RESULT_SAVED_PATH, f'j_MG_' + 'idx-' + str(initial_idx.item()) + '_test.npy')
                np.save(j_MG_path, j_MG_memory)

                time_MG_path = os.path.join(RESULT_SAVED_PATH, f'time_MG_' + 'idx-' + str(initial_idx.item()) + '_test.npy')
                np.save(time_MG_path, time_MG_memory)

            # normal simulation
            # ini_joint_states, ini_x_states, ini_mpc_cost, ini_joint_inputs, ini_abs_distance, x_collecting_ini, u_collecting_ini = mpc.MG_simulate(initial_guess,initial_state,initial_idx)
            #new_joint_states = np.array(new_joint_states)
            print(f'index {initial_idx.item()} initial data control loop & saving finished!!!!!!')
            print(f'x_data size -- {x_collecting_ini.shape}')
            print(f'u_data size -- {u_collecting_ini.shape}')
            print(f'----------------------------------------------------')

            ############################## j and time saving ##############################


            # u_ini_memory[CONTROL_STEPS*(initial_idx.item()):CONTROL_STEPS*(initial_idx.item()) + CONTROL_STEPS, :, :] = u_collecting_ini
            # x_ini_memory[CONTROL_STEPS*(initial_idx.item()):CONTROL_STEPS*(initial_idx.item()) + CONTROL_STEPS, :] = x_collecting_ini.reshape(CONTROL_STEPS,20)
            # j_ini_memory[CONTROL_STEPS*(initial_idx.item()):CONTROL_STEPS*(initial_idx.item()) + CONTROL_STEPS, :] =  np.array(ini_mpc_cost).reshape(CONTROL_STEPS,1)



            ###################### Plot results ######################
            ts = SAMPLING_TIME # 0.005
            n = len(mpc_cost)
            # print(f'mpc_cost -- {mpc_cost}')
            # print(f'n -- {n}')
            t = np.arange(0, n*ts, ts) # np.arange(len(joint_states[1])) * panda.opt.timestep
            print(f't -- {len(t)}')


            # plt 1 3d figure
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(x_states[1], x_states[2], x_states[3])

            # final & target point
            point = [x_states[1][-1], x_states[2][-1], x_states[3][-1]]
            ax.scatter(point[0], point[1], point[2], color='green', s=10)
            
            target = TARGET_POS
            ax.scatter(target[0], target[1], target[2], color='red', s=10)

            # Set axis labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim([-1.5, 1.5])  # x-axis range
            ax.set_ylim([-1.5, 1.5])  # y-axis range
            ax.set_zlim([-1.5, 1.5])   # z-axis range
            ax.legend()

            figure_name = 'idx-' + str(initial_idx.item()) + '_' + str(initial_state) + '_3d' + '.png'
            figure_path = os.path.join(RESULT_SAVED_PATH, figure_name)
            plt.savefig(figure_path)

            # plot 2 joint space (7 joints)
            plt.figure()
            for i in range(7):
                  plt.plot(t, joint_states[i+1], label=f"Joint {i+1}")
                  
            plt.xlabel("Time [s]")
            plt.ylabel("Joint position [rad]")
            plt.legend()

            figure_name = 'idx-' + str(initial_idx.item()) + '_' + str(initial_state) + '_joints' + '.png'
            figure_path = os.path.join(RESULT_SAVED_PATH, figure_name)
            plt.savefig(figure_path)

            # plot 3 distance cost
            plt.figure()
            plt.plot(t, mpc_cost, color = 'red')
            plt.xlabel("Time [s]")
            plt.ylabel("mpc cost")

            figure_name = 'idx-' + str(initial_idx.item()) + '_' + str(initial_state) + '_cost' + '.png'
            figure_path = os.path.join(RESULT_SAVED_PATH, figure_name)
            plt.savefig(figure_path)

            # plot 4 u trajectory
            plt.figure()
            for i in range(7):
                  plt.plot(t, joint_inputs[i+1], label=f"Joint {i+1}")

            plt.xlabel("Time [s]")
            plt.ylabel("Joint Control Inputs")
            plt.legend()

            figure_name = 'idx-' + str(initial_idx.item()) + '_' + str(initial_state) + '_u' + '.png'
            figure_path = os.path.join(RESULT_SAVED_PATH, figure_name)
            plt.savefig(figure_path)

            # plot 5 absolute distance
            plt.figure()
            plt.plot(t, abs_distance)

            plt.xlabel("Time [s]")
            plt.ylabel("absolute distance [m]")

            figure_name = 'idx-' + str(initial_idx.item()) + '_' + str(initial_state) + '_dis' + '.png'
            figure_path = os.path.join(RESULT_SAVED_PATH, figure_name)
            plt.savefig(figure_path)
      except Exception as e:
            print("wrong!")
            # print("fail group:", "[",initial_idx,",",n,",",ctl_step,"]")
            print(f"Error: {e}")




if __name__ == "__main__":
      main()