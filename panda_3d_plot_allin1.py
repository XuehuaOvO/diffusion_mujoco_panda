import mujoco
import numpy as np
import os
import torch

from mpc_mujoco.model import MPC
from mpc_mujoco.data_collecting_model import Cartesian_Collecting_MPC
from mpc_mujoco.cartesian_model import Cartesian_MPC
from mpc_mujoco.joint_model import Joint_MPC

import matplotlib.pyplot as plt

TARGET_POS = np.array([0.3, 0.3, 0.5])
U_INI_GUESS = '0_0_0_0_0_0_0'
HORIZON = 128
SAMPLING_TIME = 0.001
SUM_CTL_STEPS = 200
# RESULTS_DIR = '/root/diffusion_mujoco_panda/results/10_1_10'
FOLDER_PATH = '/root/diffusion_mujoco_panda/data_collecting_results/collecting_test' 

if __name__ == "__main__":
    panda = mujoco.MjModel.from_xml_path('/root/diffusion_mujoco_panda/xml/mjx_scene.xml')
    data = mujoco.MjData(panda)
    for body_id in range(panda.nbody):  # nbody is the total number of bodies
        body_name = mujoco.mj_id2name(panda, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"Body ID {body_id}: {body_name}")

    np.random.seed(42)

    # define the u initial guess range
    # u_ini_guess_range = np.linspace(-2,2,41)
    joint4_ini_guess_range = np.linspace(-1,1,2)
    # joint5_ini_guess_range = np.linspace(-1,1,2)
    joint7_ini_guess_range = np.linspace(-1,1,2)
    u_ini_guess = []
    for a in joint4_ini_guess_range:
          for c in joint7_ini_guess_range:
               u_ini_guess.append([a,c])
    u_ini_guess = np.array(u_ini_guess)
    u_ini_guess = u_ini_guess[0:3,:]
    # num_random_ini_states_for_one_u_guess = 1
    horizon = HORIZON 
    control_steps = SUM_CTL_STEPS

    num_data = len(u_ini_guess)

    # initial setting
    joints_ini_setting = np.zeros([num_data,9])
    joints_ini_setting[0:len(u_ini_guess),0] = u_ini_guess[:,0]
    # joints_ini_setting[0:len(u_ini_guess),1] = u_ini_guess[:,1]
    joints_ini_setting[0:len(u_ini_guess),2] = u_ini_guess[:,1]

    # 3d states memory
    x_states_memory =  np.zeros([num_data,control_steps,3])
    # random_starting_pos = len(u_ini_guess)

    # for i in range(len(u_ini_guess)):
    #      joints_ini_setting[random_starting_pos:random_starting_pos+num_random_ini_states_for_one_u_guess,0:3] = u_ini_guess[i,:]
    #      random_starting_pos += num_random_ini_states_for_one_u_guess
   
    # random initial state 
    # define initial state via Gaussian noise (mean = 0, sd = 0.1)
    # mean = 0          # Mean of the distribution
    # std_dev = 0.1     # Standard deviation of the distribution
    
    # random_list = []
    # for i in range(len(u_ini_guess)*num_random_ini_states_for_one_u_guess):
    #         gaussian_noise_1 = np.round(np.random.normal(mean, std_dev),2)
    #         gaussian_noise_2 = np.round(np.random.normal(mean, std_dev),2)
    #         gaussian_noise_3 = np.round(np.random.normal(mean, std_dev),2)
    #         gaussian_noise_4 = np.round(np.random.normal(mean, std_dev),2)
    #         gaussian_noise_5 = np.round(np.random.normal(mean, std_dev),2)
    #         gaussian_noise_6 = np.round(np.random.normal(mean, std_dev),2)
    #         random_list.append([gaussian_noise_1,gaussian_noise_2,gaussian_noise_3,gaussian_noise_4,gaussian_noise_5,gaussian_noise_6])

    # random_ini_states = np.array(random_list)

    # joints_ini_setting[len(u_ini_guess):,3] = random_ini_states[:,0]
    # joints_ini_setting[len(u_ini_guess):,4] = random_ini_states[:,1]
    # joints_ini_setting[len(u_ini_guess):,5] = random_ini_states[:,2]
    # joints_ini_setting[len(u_ini_guess):,6] = random_ini_states[:,3]
    # joints_ini_setting[len(u_ini_guess):,7] = random_ini_states[:,4]
    # joints_ini_setting[len(u_ini_guess):,8] = random_ini_states[:,5]

    # initial_state = [gaussian_noise_1, gaussian_noise_2, gaussian_noise_3, gaussian_noise_4, gaussian_noise_5, gaussian_noise_6]
    # initial_state = [0,0,0,0,0,0]
    # data.qpos[:6] = initial_state # different joints initial states

    # print(f'initial q_pos -- {np.array(data.qpos).reshape(-1, 1)}')
    # print(f'initial x_pos -- {np.array(data.xpos)}')
    # viewer = mujoco.viewer.launch(panda, data)
    
    # data collecting torch
    data_groups = len(u_ini_guess)*control_steps # + len(u_ini_guess)*num_random_ini_states_for_one_u_guess*control_steps
    x0_tensor = torch.zeros(data_groups,6) # 3*x + 3*x_dot
    u_sequence_tensor = torch.zeros(data_groups, horizon, 7) # 7 joints u
    
    
    # data collecting loop
    # collecting_starting_pos = 0
    for n in range(len(joints_ini_setting)):
         u_initial_guess = joints_ini_setting[n,0:3]
         initial_state = joints_ini_setting[n,3:9]
         print(f'u_initial_guess -- {u_initial_guess}')
         print(f'initial_state -- {initial_state}')
         data.qpos[:6] = initial_state 

         mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

         joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x0_collecting_1_setting, u_collecting_1_setting = mpc.simulate(u_initial_guess)
         x_states_memory[n,:,:] = x_states
         

        #  x0_1_setting_collecting = torch.tensor(x0_collecting_1_setting)
        #  u_1_setting_collecting = torch.tensor(u_collecting_1_setting)

        #  u_sequence_tensor[collecting_starting_pos:collecting_starting_pos+control_steps,:,:] = u_1_setting_collecting
        #  x0_tensor[collecting_starting_pos:collecting_starting_pos+control_steps,:] = x0_1_setting_collecting 

    ###################### Plot results ######################
    ts = SAMPLING_TIME # 0.005
    n = len(mpc_cost)
    # print(f'mpc_cost -- {mpc_cost}')
    print(f'n -- {n}')
    t = np.arange(0, n*ts, ts) # np.arange(len(joint_states[1])) * panda.opt.timestep
    print(f't -- {len(t)}')


    # plt 1 3d figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.plot(x_states[1], x_states[2], x_states[3])
    for n in range(num_data):
         x_states = x_states_memory[n,:,:]
         ax.plot(x_states[:,0], x_states[:,1], x_states[:,2])
         point = [x_states[-1,0], x_states[-1,1], x_states[-1,2]]
         ax.scatter(point[0], point[1], point[2], color='green', s=10)

    # final & target point

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

    figure_name = str(u_initial_guess) + '_' + str(initial_state) + '_3d' + '_allin1' + '.pdf'
    figure_path = os.path.join(FOLDER_PATH, figure_name)
    plt.savefig(figure_path)

    figure_name = str(u_initial_guess) + '_' + str(initial_state) + '_3d' + '_allin1' + '.png'
    figure_path = os.path.join(FOLDER_PATH, figure_name)
    plt.savefig(figure_path)
    plt.show()

    # collecting_starting_pos = collecting_starting_pos + control_steps

        #  # plot 2 joint space (7 joints)
        #  plt.figure()
        #  for i in range(7):
        #      plt.plot(t, joint_states[i + 1], label=f"Joint {i+1}")
        #  plt.xlabel("Time [s]")
        #  plt.ylabel("Joint position [rad]")
        #  plt.legend()

        #  figure_name = str(u_initial_guess)  + '_' + str(initial_state) + '_joints' + '.png'
        #  figure_path = os.path.join(FOLDER_PATH, figure_name)
        #  plt.savefig(figure_path)

        #  # plot 3 distance cost
        #  plt.figure()
        #  plt.plot(t, mpc_cost)
        #  plt.xlabel("Time [s]")
        #  plt.ylabel("mpc cost")

        #  figure_name = str(u_initial_guess)  + '_' + str(initial_state) + '_cost' + '.png'
        #  figure_path = os.path.join(FOLDER_PATH, figure_name)
        #  plt.savefig(figure_path)

        #  # plot 4 u trajectory
        #  plt.figure()
        #  for i in range(7):
        #      plt.plot(t, joint_inputs[i + 1], label=f"Joint {i+1}")
        #  plt.xlabel("Time [s]")
        #  plt.ylabel("Joint Control Inputs")
        #  plt.legend()

        #  figure_name = str(u_initial_guess) + '_' + str(initial_state) + '_u' + '.png'
        #  figure_path = os.path.join(FOLDER_PATH, figure_name)
        #  plt.savefig(figure_path)

        #  # plot 5 absolute distance
        #  plt.figure()
        #  plt.plot(t, abs_distance)
        #  plt.xlabel("Time [s]")
        #  plt.ylabel("absolute distance [m]")

        #  figure_name = str(u_initial_guess)  + '_' + str(initial_state) + '_dis' + '.png'
        #  figure_path = os.path.join(FOLDER_PATH, figure_name)
        #  plt.savefig(figure_path)

        # collecting_starting_pos = collecting_starting_pos + control_steps

# save u data in PT file for training
# torch.save(u_sequence_tensor, os.path.join(FOLDER_PATH , f'u_sequence_tensor_'+ str(data_groups)+'-8-1_test.pt'))

# # save x0 data in PT file as conditional info in training
# torch.save(x0_tensor, os.path.join(FOLDER_PATH , f'x0_tensor_'+ str(data_groups)+'-4_test.pt'))
