import mujoco
# from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np

from mpc_mujoco.model import MPC
from mpc_mujoco.cartesian_model import Cartesian_MPC
from mpc_mujoco.joint_model import Joint_MPC

import matplotlib.pyplot as plt

TARGET_POS = np.array([0.4, 0.4, 0.3])

if __name__ == "__main__":
    # panda = load_robot_description("panda_mj_description")
    panda = mujoco.MjModel.from_xml_path('/home/xiao/diffusion_mujoco/xml/mjx_scene.xml')
    data = mujoco.MjData(panda)
    for body_id in range(panda.nbody):  # nbody is the total number of bodies
        body_name = mujoco.mj_id2name(panda, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"Body ID {body_id}: {body_name}")
    # data.qpos[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    # initial_qpos = np.array(data.qpos).reshape(-1, 1)
    # mujoco.mj_fwdPosition(panda, data) # also mujoco.mj_invPosition!!!!!!!
    print(f'initial q_pos -- {np.array(data.qpos).reshape(-1, 1)}')
    # print(f'initial q_vel -- {np.array(data.qvel).reshape(-1, 1)}')
    print(f'initial x_pos -- {np.array(data.xpos)}')
    # viewer = mujoco.viewer.launch(panda, data)
    
    # mpc = MPC(data=data, trajectory_id=0)
    mpc = Cartesian_MPC(panda = panda, data=data)
    # mpc = Joint_MPC(panda = panda, data=data)

    # Plot results
    # joint_states, x_states, mpc_cost = mpc.simulate()
    joint_states, x_states, mpc_cost, joint_inputs = mpc.simulate()

    ts = 0.01
    n = len(x_states[1])
    t = np.arange(0, n*ts, ts) # np.arange(len(joint_states[1])) * panda.opt.timestep

    # plt.figure()
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
    r = 0.15  # Radius of the sphere
    pi = np.pi
    cos = np.cos
    sin = np.sin

    # Define spherical coordinates
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    # Define the center of the sphere
    center = [0, 0, 0.7]

    # Shift the sphere by the center
    x += center[0]
    y += center[1]
    z += center[2]

    # Plot the surface of the sphere
    ax.plot_surface(x, y, z, color='blue', alpha=0.6)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([-1.5, 1.5])  # x-axis range
    ax.set_ylim([-1.5, 1.5])  # y-axis range
    ax.set_zlim([-1.5, 1.5])   # z-axis range
    ax.legend()

    # plot 2 joint space (7 joints)
    plt.figure()
    for i in range(7):
        plt.plot(t, joint_states[i + 1], label=f"Joint {i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.legend()

    # plot 3 distance cost
    plt.figure()
    plt.plot(t, mpc_cost)
    plt.xlabel("Time [s]")
    plt.ylabel("mpc cost")
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot()

    # plot 4 u trajectory
    plt.figure()
    for i in range(7):
        plt.plot(t, joint_inputs[i + 1], label=f"Joint {i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint Control Inputs")
    plt.legend()

    # for i in range(7):
    #     ax.plot(t, joint_states[i + 1], label=f"Joint {i+1}")
    # ax2.set_xlabel("Time [s]")
    # ax2.set_ylabel("Joint position [rad]")
    # ax2.legend()
    # plt.show()

    # for i in range(3):
    #     plt.plot(t, x_states[i + 1], label=f"xpos {i+1}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("X position")

    # plt.legend()
    plt.show()