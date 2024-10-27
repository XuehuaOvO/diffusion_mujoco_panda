import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
from mpc_mujoco.model import MPC
import matplotlib.pyplot as plt
from pyquaternion import Quaternion as pyq


def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def main():
    panda = load_robot_description("panda_mj_description")
    data = mujoco.MjData(panda)
    # data.qpos[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    # initial_qpos = np.array(data.qpos).reshape(-1, 1)
    # mujoco.mj_fwdPosition(panda, data) # also mujoco.mj_invPosition!!!!!!!
    print(f'initial q_pos -- {np.array(data.qpos).reshape(-1, 1)}')
    # print(f'initial q_vel -- {np.array(data.qvel).reshape(-1, 1)}')
    print(f'initial x_pos -- {np.array(data.xpos)}')
    # viewer = mujoco.viewer.launch(panda, data)
    # Run the viewer to display the initial state of the robot
    with mujoco.viewer.launch_passive(panda, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(panda, data)

            viewer.sync()


if __name__ == "__main__":
    main()