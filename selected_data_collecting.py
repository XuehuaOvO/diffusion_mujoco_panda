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
CONTROL_STEPS = 200 # 200 ########
HOR = 128
NUM_SEED = 42 # 42
MAX_CORE_CPU = 5 # 5 ##########
SAMPLING_TIME = 0.001
TARGET_POS = np.array([0.3, 0.3, 0.5])

NUM_INI_STATES = 5 # 5 #########
NOISE_DATA_PER_STATE = 4 # 4 ###
U_GUESS_NUMBER = 5

# data saving
RESULT_SAVED_PATH = '/root/diffusion_mujoco_panda/selected_data'

def main():
    # memories
    j_SG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200
    time_SG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200
    j_MG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200
    time_MG_memory = np.zeros((1, CONTROL_STEPS)) # 1*200

    # memories for data
    u_ini_memory = np.zeros((1*CONTROL_STEPS, 128, 7)) # 1*200, 128, 7 
    u_random_memory = np.zeros((NOISE_DATA_PER_STATE*CONTROL_STEPS, 128, 7))# 4*200, 128, 7 
    x_ini_memory = np.zeros((1*CONTROL_STEPS, 20)) # 1*200, 20 (q q_dot x x_dot) 
    x_random_memory = np.zeros((NOISE_DATA_PER_STATE*CONTROL_STEPS, 20)) # 4*200, 20 (q q_dot x x_dot) 
    j_ini_memory = np.zeros((1*CONTROL_STEPS, 1)) # 1*200, 1 
    j_random_memory = np.zeros((NOISE_DATA_PER_STATE*CONTROL_STEPS, 1)) # 4*200, 1 


    # initial states generating
    ini_states, random_ini_u_guess, ini_data_idx = ini_states_idx_generating()

    # initial data groups
    initial_data_groups = [] # SG 10 + MG 10
    for n in range(NUM_INI_STATES):
          initial_data_groups.append([ini_states[n,:], random_ini_u_guess[n:n+U_GUESS_NUMBER,:], ini_data_idx[n], u_ini_memory, u_random_memory, x_ini_memory, x_random_memory, j_ini_memory, j_random_memory])
    
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
    for i in range(NUM_INI_STATES):
            gaussian_noise_ini_joint_1 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_2 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_3 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_4 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_5 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_6 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            random_ini_states_list.append([gaussian_noise_ini_joint_1,gaussian_noise_ini_joint_2,gaussian_noise_ini_joint_3,gaussian_noise_ini_joint_4,gaussian_noise_ini_joint_5,gaussian_noise_ini_joint_6, 0])   
    random_ini_states = np.array(random_ini_states_list) # 20*7 

    # initial u guess
    random_u_guess_list = []
    for i in range(NUM_INI_STATES):
          for j in range(U_GUESS_NUMBER):
                u_guess_4 = np.round(random.uniform(-2, 2),2)
                u_guess_5 = np.round(random.uniform(-2, 2),2)
                u_guess_7 = np.round(random.uniform(-2, 2),2)
                random_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])
    random_ini_u_guess = np.array(random_u_guess_list) # 5*5, 7

    # data idx
    idx_list = []
    for i in range(NUM_INI_STATES):
          idx = i
          idx_list.append(idx)
    data_idx = np.array(idx_list)

    return random_ini_states, random_ini_u_guess,  data_idx

# def SG_ini_u_guess_generating():
#       # random initial u guess
#       random_u_guess_list = []

#       u_guess_4 = np.round(random.uniform(-2, 2),2)
#       u_guess_5 = np.round(random.uniform(-2, 2),2)
#       u_guess_7 = np.round(random.uniform(-2, 2),2)
#       random_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])

#       random_SG_ini_u_guess = np.array(random_u_guess_list) # 1*7
#       return random_SG_ini_u_guess

# def MG_ini_u_guess_generating():
#       # random initial u guess
#       random_u_guess_list = []
#       for i in range(MG_U_GUESS_NUMBER):
#             u_guess_4 = np.round(random.uniform(-2, 2),2)
#             u_guess_5 = np.round(random.uniform(-2, 2),2)
#             u_guess_7 = np.round(random.uniform(-2, 2),2)
#             random_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])

#       random_MG_ini_u_guess = np.array(random_u_guess_list) # 5*7
#       return random_MG_ini_u_guess


def states_noise_generating(current_states,current_inputs):
    np.random.seed(NUM_SEED)

    # add Gaussian noise to the random initial states
    mean_noise = 0
    std_dev_noise = 0.05

    states_noise_array = np.zeros((NOISE_DATA_PER_STATE, 7))
    for i in range(NOISE_DATA_PER_STATE):
          noise_to_ini_states =  np.round(np.random.normal(mean_noise, std_dev_noise, 7),4)
          states_noise_array[i,:] = noise_to_ini_states
    
    noisy_data = np.zeros((NOISE_DATA_PER_STATE, 7)) # 4*7
    for k in range(NOISE_DATA_PER_STATE):
          noisy_data[k,:] = current_states + states_noise_array[k,:]

    noisy_data_u_guess_list = []
    for i in range(NOISE_DATA_PER_STATE):
          for num in range(U_GUESS_NUMBER):
               u_noisy_guess_4 = np.round(random.uniform(current_inputs[num,3]-2, current_inputs[num,3]+2),2)
               u_noisy_guess_5 = np.round(random.uniform(current_inputs[num,4]-2, current_inputs[num,4]+2),2)
               u_noisy_guess_7 = np.round(random.uniform(current_inputs[num,6]-2, current_inputs[num,6]+2),2)
               noisy_data_u_guess_list.append([current_inputs[num,0],current_inputs[num,1],current_inputs[num,2],u_noisy_guess_4,u_noisy_guess_5,current_inputs[num,5],u_noisy_guess_7])
    noisy_data_u_guess =  np.array(noisy_data_u_guess_list) # 4*5, 7
          
    return states_noise_array, noisy_data, noisy_data_u_guess


# def next_noise_guesses_generating(current_inputs):
#     np.random.seed(NUM_SEED)

#     next_u_MG_guess = []

#     for num in range(U_GUESS_NUMBER):
#         u_noisy_guess_4 = np.round(random.uniform(current_inputs[3]-2, current_inputs[3]+2),2)
#         u_noisy_guess_5 = np.round(random.uniform(current_inputs[4]-2, current_inputs[4]+2),2)
#         u_noisy_guess_7 = np.round(random.uniform(current_inputs[6]-2, current_inputs[6]+2),2)
#         next_u_MG_guess.append([current_inputs[0],current_inputs[1],current_inputs[2],u_noisy_guess_4,u_noisy_guess_5,current_inputs[5],u_noisy_guess_7])
#     next_u_MG_guess = np.array(next_u_MG_guess)
    
#     return next_u_MG_guess


def single_mpc_process(initial_state, random_ini_u_guess, initial_idx, u_ini_memory, u_random_memory, x_ini_memory, x_random_memory, j_ini_memory, j_random_memory):
      try:
            # panda mujoco
            panda = mujoco.MjModel.from_xml_path('/root/diffusion_mujoco_panda/xml/mjx_scene.xml')
            data = mujoco.MjData(panda)
            mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

            # Simulation
            joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x_collecting_ini, u_collecting_ini, delta_t = mpc.MG_simulate(random_ini_u_guess,initial_state,initial_idx)

            # j_MG_memory = np.array(mpc_cost).reshape(1,CONTROL_STEPS)
            # time_MG_memory = np.array(delta_t).reshape(1,CONTROL_STEPS)

            # j_MG_path = os.path.join(RESULT_SAVED_PATH, f'j_MG_' + 'idx-' + str(initial_idx.item()) + '_test.npy')
            # np.save(j_MG_path, j_MG_memory)

            # time_MG_path = os.path.join(RESULT_SAVED_PATH, f'time_MG_' + 'idx-' + str(initial_idx.item()) + '_test.npy')
            # np.save(time_MG_path, time_MG_memory)

            # normal simulation
            # ini_joint_states, ini_x_states, ini_mpc_cost, ini_joint_inputs, ini_abs_distance, x_collecting_ini, u_collecting_ini = mpc.MG_simulate(initial_guess,initial_state,initial_idx)
            #new_joint_states = np.array(new_joint_states)
            print(f'index {initial_idx.item()} initial data control loop finished!!!!!!')
            print(f'x_data size -- {x_collecting_ini.shape}')
            print(f'u_data size -- {u_collecting_ini.shape}')
            print(f'----------------------------------------------------')

            u_ini_memory = u_collecting_ini
            x_ini_memory = x_collecting_ini.reshape(CONTROL_STEPS,20)
            j_ini_memory =  np.array(mpc_cost).reshape(CONTROL_STEPS,1)

            noi_joint_states = np.zeros([CONTROL_STEPS,NOISE_DATA_PER_STATE,7])
            noi_joint_inputs = np.zeros([CONTROL_STEPS,NOISE_DATA_PER_STATE,7])
            noi_mpc_cost = np.zeros([CONTROL_STEPS,NOISE_DATA_PER_STATE,1])
            noi_abs_distance = np.zeros([CONTROL_STEPS,NOISE_DATA_PER_STATE,1])

            ############################################## data with noise ##############################################

            for ctl_step in range(CONTROL_STEPS):
                    current_states = np.zeros(7)
                    current_inputs = np.zeros([5,7])

                    print(f'[initial_idx, ctl_step] -- {initial_idx},{ctl_step}')
                    for i in range(7):
                        current_states[i] = joint_states[i+1][ctl_step]

                    if ctl_step == 0:
                        current_inputs = random_ini_u_guess.copy()
                    else:
                        for i in range(7):
                                current_inputs[:,i] = joint_inputs[i+1][ctl_step-1]
                    # data.qpos[:7] = current_states
                    # mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

                    # noisy data generating
                    noise_array, noisy_states, noisy_data_u_guess = states_noise_generating(current_states,current_inputs) # 20*7

                    # print(f'current_states -- {current_states}')

                    # standard u generating
                    # ini_joint_states, ini_x_states, ini_mpc_cost, ini_joint_inputs, ini_abs_distance, x_collecting_1_step, u_collecting_1_step = mpc.simulate(initial_guess)
                    # #new_joint_states = np.array(new_joint_states)
                    # print(f'index {initial_idx.item()} initial data control loop finished!!!!!!')
                    # # print(f'x_data size -- {x_collecting_1_step.shape}')
                    # # print(f'u_data size -- {u_collecting_1_step.shape}')
                    # print(f'----------------------------------------------------')
                    
                    # plotting data append
                    # std_joint_states.append(ini_joint_states)
                    # std_joint_inputs.append(ini_joint_inputs)
                    # std_x_states.append(ini_x_states)
                    # std_mpc_cost.append(ini_mpc_cost)
                    # std_abs_distance.append(ini_abs_distance)

                    # print(f'idx -- {initial_idx.item()}')
                    # print(f'j -- {ini_mpc_cost.item()}')
                    # u_ini_memory[200*(initial_idx.item()) + ctl_step, :, :] = u_collecting_1_step
                    # x_ini_memory[200*(initial_idx.item()) + ctl_step, :] = x_collecting_1_step.reshape(1,20)
                    # j_ini_memory[200*(initial_idx.item()) + ctl_step, :] =  np.array(ini_mpc_cost).reshape(1,1)
                    
                    # data with noise

                    # noise_data_starting_pos = (initial_idx.item())*CONTROL_STEPS*NOISE_DATA_PER_STATE
                    for n in range(NOISE_DATA_PER_STATE):
                        noisy_state_n = noisy_states[n,:]
                        # data.qpos[:7] = noisy_state
                        # mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

                        # each selected data has 5 guesses
                        noisy_u_guess = noisy_data_u_guess[n:n+U_GUESS_NUMBER,:]

                        # print(f'noisy_state -- {noisy_state_n}')
                        # print(f'noisy_u_guess -- {noisy_u_guess}')

                        random_joint_states, random_x_states, random_mpc_cost, random_joint_inputs, random_abs_distance, x_noisy_collecting_1_step, u_noisy_collecting_1_step, new_noisy_joint_states = mpc.selelcted_noise_data_simulate(noisy_u_guess,noisy_state_n,initial_idx, ctl_step, n)

                        # noi_joint_states.append(random_joint_states)
                        # noi_joint_inputs.append(random_joint_inputs)
                        # # noi_x_states.append(random_x_states)
                        # noi_mpc_cost.append(random_mpc_cost)
                        # noi_abs_distance.append(random_abs_distance)

                        noi_joint_states[ctl_step,n,:] = random_joint_states.reshape(7)
                        noi_joint_inputs[ctl_step,n,:] = random_joint_inputs.reshape(7)
                        noi_mpc_cost[ctl_step,n,0] = random_mpc_cost
                        noi_abs_distance[ctl_step,n,0] = random_abs_distance
                        
                        # save noisy data
                        print(f'location -- {n * CONTROL_STEPS + ctl_step}')
                        u_random_memory[n * CONTROL_STEPS + ctl_step, :, :] = u_noisy_collecting_1_step
                        x_random_memory[n * CONTROL_STEPS + ctl_step, :] = x_noisy_collecting_1_step.reshape(1,20)
                        j_random_memory[n * CONTROL_STEPS + ctl_step, 0] = np.array(random_mpc_cost).reshape(1,1)

                        # print(f'n --{n}')
            
            #################### data saving
            # to tensor
            torch_u_ini_memory_tensor = torch.Tensor(u_ini_memory)
            torch_u_random_memory_tensor = torch.Tensor(u_random_memory)
            torch_x_ini_memory_tensor = torch.Tensor(x_ini_memory)
            torch_x_random_memory_tensor = torch.Tensor(x_random_memory)
            torch_j_ini_memory_tensor = torch.Tensor(j_ini_memory)
            torch_j_random_memory_tensor = torch.Tensor(j_random_memory)
            
            # cat
            u_data = torch.cat((torch_u_ini_memory_tensor, torch_u_random_memory_tensor), dim=0)
            x_data = torch.cat((torch_x_ini_memory_tensor, torch_x_random_memory_tensor), dim=0)
            j_data = torch.cat((torch_j_ini_memory_tensor, torch_j_random_memory_tensor), dim=0)

            # save data in PT file for training
            torch.save(u_data, os.path.join(RESULT_SAVED_PATH , f'u_data_' + 'idx-' + str(initial_idx.item()) + '_selected_test.pt'))
            torch.save(x_data, os.path.join(RESULT_SAVED_PATH , f'x_data_' + 'idx-' + str(initial_idx.item()) + '_selected_test.pt'))
            torch.save(j_data, os.path.join(RESULT_SAVED_PATH , f'j_data_' + 'idx-' + str(initial_idx.item()) + '_selected_test.pt'))

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
            for z in range(CONTROL_STEPS):
                  for i in range(7):
                        noisy_q_each_ctl_step = noi_joint_states[z,:,i]
                        for k in range(NOISE_DATA_PER_STATE):
                              plt.scatter(t[z], noisy_q_each_ctl_step[k], color = 'blue')      
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
            for z in range(CONTROL_STEPS):
                  noisy_cost_each_ctl_step = noi_mpc_cost[z,:,:]
                  for k in range(NOISE_DATA_PER_STATE):
                        plt.scatter(t[z], noisy_cost_each_ctl_step[k], color = 'blue')

            figure_name = 'idx-' + str(initial_idx.item()) + '_' + str(initial_state) + '_cost' + '.png'
            figure_path = os.path.join(RESULT_SAVED_PATH, figure_name)
            plt.savefig(figure_path)

            # plot 4 u trajectory
            plt.figure()
            for i in range(7):
                  plt.plot(t, joint_inputs[i+1], label=f"Joint {i+1}")
            for z in range(CONTROL_STEPS):
                  for i in range(7):
                        noisy_u_each_ctl_step = noi_joint_inputs[z,:,i]
                        for k in range(NOISE_DATA_PER_STATE):
                              plt.scatter(t[z], noisy_u_each_ctl_step[k], color = 'blue')

            plt.xlabel("Time [s]")
            plt.ylabel("Joint Control Inputs")
            plt.legend()

            figure_name = 'idx-' + str(initial_idx.item()) + '_' + str(initial_state) + '_u' + '.png'
            figure_path = os.path.join(RESULT_SAVED_PATH, figure_name)
            plt.savefig(figure_path)

            # plot 5 absolute distance
            plt.figure()
            plt.plot(t, abs_distance)
            for z in range(CONTROL_STEPS):
                  noisy_dis_each_ctl_step = noi_abs_distance[z,:,:]
                  for k in range(NOISE_DATA_PER_STATE):
                        plt.scatter(t[z], noisy_dis_each_ctl_step[k], color = 'blue')
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