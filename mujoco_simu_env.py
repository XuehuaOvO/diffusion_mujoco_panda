import mujoco
# from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import os

from mpc_mujoco.model import MPC
from mpc_mujoco.cartesian_model import Cartesian_MPC
from mpc_mujoco.joint_model import Joint_MPC

import matplotlib.pyplot as plt

TARGET_POS = np.array([0.3, 0.3, 0.5])
U_INI_GUESS = '0_0_0_-2_-2_0_-2'
RESULTS_DIR = '/root/diffusion_mujoco_panda/results/discrete_dy'

if __name__ == "__main__":
    # panda = load_robot_description("panda_mj_description")
    panda = mujoco.MjModel.from_xml_path('/root/diffusion_mujoco_panda/xml/mjx_scene.xml')
    data = mujoco.MjData(panda)
    for body_id in range(panda.nbody):  # nbody is the total number of bodies
        body_name = mujoco.mj_id2name(panda, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"Body ID {body_id}: {body_name}")
    

    # initial_state = [gaussian_noise_1, gaussian_noise_2, gaussian_noise_3, gaussian_noise_4, gaussian_noise_5, gaussian_noise_6]
    initial_state = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    # initial_state = [ 0, 0, 0, 0, 0, 0]
    data.qpos[:7] = initial_state # different joints initial states
    data.qvel[:] = 0  # Set all joint velocities to zero
    data.qacc[0:7] = 0  # Set all joint accelerations to zero
    print("qacc:", data.qacc[:7])
    data.ctrl[:] = 0 
    
    #mujoco.mj_forward(panda, data)
    # mujoco.mj_step(panda, data)

    print(f'initial q_pos -- {np.array(data.qpos).reshape(-1, 1)}')
    print(f'initial x_pos -- {np.array(data.xpos)}')
    print("qvel:", data.qvel[:7])
    print("qacc:", data.qacc[:7])
    viewer = mujoco.viewer.launch(panda, data)
    viewer._paused = True  # Manually pause the simulation