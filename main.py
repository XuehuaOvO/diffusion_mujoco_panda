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
    
    # define initial state via Gaussian noise (mean = 0, sd = 0.1)
    mean = 0          # Mean of the distribution
    std_dev = 0.1     # Standard deviation of the distribution
    gaussian_noise_1 = np.round(np.random.normal(mean, std_dev),2)
    gaussian_noise_2 = np.round(np.random.normal(mean, std_dev),2)
    gaussian_noise_3 = np.round(np.random.normal(mean, std_dev),2)
    gaussian_noise_4 = np.round(np.random.normal(mean, std_dev),2)
    gaussian_noise_5 = np.round(np.random.normal(mean, std_dev),2)
    gaussian_noise_6 = np.round(np.random.normal(mean, std_dev),2)

    # initial_state = [gaussian_noise_1, gaussian_noise_2, gaussian_noise_3, gaussian_noise_4, gaussian_noise_5, gaussian_noise_6]
    # initial_state = [ 0.05, -0.01,  0.06,  0.15, -0.02, -0.02]
    initial_state = [ 0, 0, 0, 0, 0, 0]
    data.qpos[:6] = initial_state # different joints initial states

    print(f'initial q_pos -- {np.array(data.qpos).reshape(-1, 1)}')
    print(f'initial x_pos -- {np.array(data.xpos)}')
    # viewer = mujoco.viewer.launch(panda, data)
    
    # mpc = MPC(data=data, trajectory_id=0)
    mpc = Cartesian_MPC(panda = panda, data=data)
    # mpc = Joint_MPC(panda = panda, data=data)

    joint_states, x_states, mpc_cost, joint_inputs, abs_distance = mpc.simulate()

    ###################### Plot results ######################
    ts = 0.001 # 0.005
    n = len(mpc_cost)
    # print(f'mpc_cost -- {mpc_cost}')
    print(f'n -- {n}')
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

    # obstacle ball     <geom name="obstacle" type="sphere" pos="0.15 0.15 0.7" size="0.1" rgba="0 0 1 0.5"/>
    # Define the center and radius of the ball
    # Sphere parameters
    # r = 0.15  # Radius of the sphere
    # pi = np.pi
    # cos = np.cos
    # sin = np.sin

    # # Define spherical coordinates
    # phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    # x = r * sin(phi) * cos(theta)
    # y = r * sin(phi) * sin(theta)
    # z = r * cos(phi)

    # # Define the center of the sphere
    # center = [0, 0, 0.7]

    # # Shift the sphere by the center
    # x += center[0]
    # y += center[1]
    # z += center[2]

    # # Plot the surface of the sphere
    # ax.plot_surface(x, y, z, color='blue', alpha=0.6)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([-1.5, 1.5])  # x-axis range
    ax.set_ylim([-1.5, 1.5])  # y-axis range
    ax.set_zlim([-1.5, 1.5])   # z-axis range
    ax.legend()

    figure_name = U_INI_GUESS + '_' + str(initial_state) + '_3d' + '.png'
    figure_path = os.path.join(RESULTS_DIR, figure_name)
    plt.savefig(figure_path)

    # plot 2 joint space (7 joints)
    plt.figure()
    for i in range(7):
        plt.plot(t, joint_states[i + 1], label=f"Joint {i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.legend()

    figure_name = U_INI_GUESS + '_' + str(initial_state) + '_joints' + '.png'
    figure_path = os.path.join(RESULTS_DIR, figure_name)
    plt.savefig(figure_path)

    # plot 3 distance cost
    plt.figure()
    plt.plot(t, mpc_cost)
    plt.xlabel("Time [s]")
    plt.ylabel("mpc cost")

    figure_name = U_INI_GUESS + '_' + str(initial_state) + '_cost' + '.png'
    figure_path = os.path.join(RESULTS_DIR, figure_name)
    plt.savefig(figure_path)

    # plot 4 u trajectory
    plt.figure()
    for i in range(7):
        plt.plot(t, joint_inputs[i + 1], label=f"Joint {i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint Control Inputs")
    plt.legend()

    figure_name = U_INI_GUESS + '_' + str(initial_state) + '_u' + '.png'
    figure_path = os.path.join(RESULTS_DIR, figure_name)
    plt.savefig(figure_path)

    # plot 5 absolute distance
    plt.figure()
    plt.plot(t, abs_distance)
    plt.xlabel("Time [s]")
    plt.ylabel("absolute distance [m]")

    figure_name = U_INI_GUESS + '_' + str(initial_state) + '_dis' + '.png'
    figure_path = os.path.join(RESULTS_DIR, figure_name)
    plt.savefig(figure_path)

    plt.show()