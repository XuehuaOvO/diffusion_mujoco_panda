import mujoco
import numpy as np
import random
import os
import torch
from multiprocessing import Pool

from mpc_mujoco.model import MPC
from mpc_mujoco.collecting_test_model import Cartesian_Collecting_MPC
from mpc_mujoco.joint_model import Joint_MPC

import matplotlib.pyplot as plt

# Setting
NUM_INI_STATES = 50
NOISE_DATA_PER_STATE = 20
CONTROL_STEPS = 200
NUM_SEED = 42
MAX_CORE_CPU = 50

def main():
    num_seed = NUM_SEED
    ini_data_start_idx = 0
    noisy_data_start_idx = NUM_INI_STATES*CONTROL_STEPS

    # initial data generating 50*7, 50*7
    random_ini_states, random_ini_u_guess = ini_data_generating() # 50*7
    
    # initial data with noise 
    # current_states = random_ini_states # 50*20*7, # 50*20*7
    # ini_noise_array, ini_noisy_states, ini_noisy_data_u_guess = states_noise_generating(current_states) 

    # initial data groups 50
    initial_data_groups = []
    for n in range(NUM_INI_STATES):
          initial_data_groups.append([random_ini_u_guess[n,:], random_ini_states[n,:]])
    
    # (noisy)
    # for a in range(NUM_INI_STATES):
    #       for b in range(NOISE_DATA_PER_STATE):
    #             initial_data_groups.append([ini_noisy_data_u_guess[a,b,:], ini_noisy_states[a,b,:]])
    
    ini_data_groups_array = np.array(initial_data_groups)
    print(f'initial_data_groups_size -- {ini_data_groups_array.shape}')

    with Pool(processes=MAX_CORE_CPU) as pool:
          pool.starmap(single_process, initial_data_groups)

    # memories 
    u_training_memory = np.zeros((NUM_INI_STATES*(1+NOISE_DATA_PER_STATE)*CONTROL_STEPS, 128, 7)) # 50*(20+1)*200 = 10000 + 200000, 128, 7
    x_conditioning_memory = np.zeros((NUM_INI_STATES*(1+NOISE_DATA_PER_STATE)*CONTROL_STEPS, 6)) # 50*(20+1)*200 = 10000 + 200000, 6 (x & x_dot)
          




##############################################################################################################

def ini_data_generating():
    np.random.seed(NUM_SEED)
    random.seed(NUM_SEED)

    # Gaussian noise for initial states generating
    mean_ini = 0
    std_dev_ini = 0.1
    
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
    random_ini_states = np.array(random_ini_states_list) # 50*7

    # random initial u guess
    u_guess_range = np.linspace(-2,2,5)
    random_u_guess_list = []
    for i in range(NUM_INI_STATES):
          selected_idx = random.randint(0, 4)
          random_u_guess_list.append([0,0,0,u_guess_range[selected_idx],u_guess_range[selected_idx],0,u_guess_range[selected_idx]])
    random_ini_u_guess = np.array(random_u_guess_list) # 50*7

    return random_ini_states, random_ini_u_guess



def states_noise_generating(current_states):
    np.random.seed(NUM_SEED)

    # add Gaussian noise to the random initial states
    mean_noise = 0
    std_dev_noise = 0.05

    states_noise_array = np.zeros((NUM_INI_STATES, NOISE_DATA_PER_STATE, 7))
    for i in range(NUM_INI_STATES):
          noise_to_ini_states =  np.random.normal(mean_noise, std_dev_noise, (NOISE_DATA_PER_STATE, 7))
          states_noise_array[i,:,:] = noise_to_ini_states
    
    noisy_data = np.zeros((NUM_INI_STATES, NOISE_DATA_PER_STATE, 7)) # 50*20*7
    for i in range(NUM_INI_STATES):
            for k in range(NOISE_DATA_PER_STATE):
                noisy_data[i,k,:] = current_states[i,:] + states_noise_array[i,k,:]

    u_guess_range = np.linspace(-2,2,5)
    noisy_data_u_guess_list = []
    for i in range(NUM_INI_STATES*NOISE_DATA_PER_STATE):
          selected_idx = random.randint(0, 4)
          noisy_data_u_guess_list.append([0,0,0,u_guess_range[selected_idx],u_guess_range[selected_idx],0,u_guess_range[selected_idx]])
    noisy_data_u_guess =  np.array(noisy_data_u_guess_list).reshape(NUM_INI_STATES,NOISE_DATA_PER_STATE,7) # 50*20*7
          
    return states_noise_array, noisy_data, noisy_data_u_guess



def single_process(initial_guess,initial_state):
      # panda mujoco
      panda = mujoco.MjModel.from_xml_path('/root/diffusion_mujoco_panda/xml/mjx_scene.xml')
      data = mujoco.MjData(panda)

      current_states = initial_state

      for ctl_step in range(CONTROL_STEPS):
            data.qpos[:7] = initial_state 
            mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

            # initial noisy data generating
            noise_array, noisy_states, noisy_data_u_guess = states_noise_generating(current_states) 

            # standard u generating
            joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x0_collecting_1_setting, u_collecting_1_setting = mpc.simulate(initial_guess)

            ############################################  data with noise  ############################################
            for c in range(len(NUM_INI_STATES)):
                  for d in range(len(NOISE_DATA_PER_STATE)):
                        noisy_state = noisy_states[c,d,:]
                        data.qpos[:7] = noisy_state
                        mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

                        noisy_state_guess = noisy_data_u_guess[c,d,:]

                        joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x0_collecting_1_setting, u_collecting_1_setting = mpc.noise_simulate(noisy_state_guess)



if __name__ == "__main__":
      main()