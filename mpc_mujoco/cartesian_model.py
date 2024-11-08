import numpy as np
import casadi as ca
import do_mpc
import mujoco
import mujoco.viewer
import time

HORIZON = 128
SUM_CTL_STEPS = 300
CONTROLLER_SAMPLE_TIME = 0.005
FIXED_TARGET = np.array([[0.3], [0.3], [0.5]]) # np.array([[0.4], [0.4], [0.3]])
TARGET_POS = np.array([0.3, 0.3, 0.5]).reshape(3, 1) # np.array([0.4, 0.4, 0.3]).reshape(3, 1)
U_INI_GUESS = ca.DM([0,0,0,0,0,0,0])

Q = np.diag([10,10,10])
R = 1
P = np.diag([10,10,10])


class Cartesian_MPC:

    def __init__(self, panda, data):
        self.panda = panda
        self.data = data
        # self.trajectory_id = trajectory_id
        mujoco.mj_forward(data.model, data)
        self.model = self.create_model()
        self.mpc = self.create_mpc(self.model)

    def get_trajectory(self, trajectory_id: int, t_now):
        if trajectory_id == 0:
            traj = np.sin(np.linspace(0, 2 * np.pi, 21)) * np.ones((7, 21))
            traj[:6, :] = 0
            t_now_scaled = t_now / 2
            k = int(t_now_scaled % 21)
            return traj.T[k]

    def get_inertia_matrix(self):
        nv = self.data.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.data.model, M, self.data.qM)
        return M[:7, :7]

    def get_coriolis(self):
        return self.data.qfrc_bias

    def get_gravity_forces(self):
        return self.data.qfrc_gravcomp
    
    # Jacobian Matrix
    def compute_jacobian(self, model, data, tpoint):
        """Compute the Jacobian for the end-effector."""
        # Jacobian matrices for position (jacp) and orientation (jacr)
        jacp = np.zeros((3, model.nv))  # Position Jacobian
        jacr = np.zeros((3, model.nv))  # Orientation Jacobian
        
        # body_name="panda hand" # panda_hand
        # body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_id = 9

        # mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        mujoco.mj_jac(model, data, jacp, jacr, tpoint, body_id)
        
        return jacp, jacr
    
    # target error
    def compute_task_space_error(self, target_pos):
        """Compute the error in task-space (end-effector position)."""
        # Get the current end-effector position (x, y, z) using MuJoCo's data
        current_pos = self.data.body("hand").xpos.copy()

        # Compute the Jacobian for the end-effector
        jacp, _ = self.compute_jacobian(self.panda, self.data, target_pos)

        # Compute the joint velocities (q_dot) from the current state
        q_dot_current = np.array(self.data.qvel[:7]).reshape(-1, 1)

        # Compute the end-effector velocity from joint velocities using the Jacobian
        # end_effector_velocity = np.dot(jacp, q_dot_current)

        # Compute the error between the current and target positions
        target = ca.DM(target_pos)
        current = ca.DM(current_pos)
        error = target - current # no orientation demand like: angular_error = target_orientation - current_orientation

        return error

    def create_model(self):
        model_type = "continuous"
        model = do_mpc.model.Model(model_type)

        # Define the states (joint positions and velocities)
        q = model.set_variable(var_type="_x", var_name="q", shape=(7, 1))
        q_dot = model.set_variable(var_type="_x", var_name="q_dot", shape=(7, 1))

        x = model.set_variable(var_type="_x", var_name="x", shape=(3, 1))
        x_dot = model.set_variable(var_type="_x", var_name="x_dot", shape=(3, 1))

        # R = model.set_variable(var_type='_u', var_name='R')

        # Position Jacobian
        jacp, _ = self.compute_jacobian(self.panda, self.data, TARGET_POS)
        jacp = jacp[:, :7]

        # Define the state target variable
        target_x_states = model.set_variable(var_type="_tvp", var_name="target_x_states", shape=(3, 1))

        # define the state initial variable
        initial_x_states = model.set_variable(var_type="_tvp", var_name="initial_x_states", shape=(3, 1))

        # Define obstacle center
        target_obs_center = model.set_variable(var_type="_tvp", var_name="target_obs_center", shape=(3, 1))

        # Define the previous control input as an auxiliary variable or parameter
        # u_prev = model.set_variable(var_type='_x', var_name="u_prev", shape=(7,1))

        # Define the control inputs (joint torques)
        tau = model.set_variable(var_type="_u", var_name="tau", shape=(7, 1))

        mujoco.mj_forward(self.data.model, self.data)  # initialize values

        M = self.get_inertia_matrix()
        C = self.get_coriolis()[:7].reshape(1, 7)
        G = self.get_gravity_forces()[:7]

        q_ddot = ca.mtimes(ca.inv(M), (tau - ca.mtimes(C, q_dot) - G))

        model.set_rhs("x",x_dot)
        model.set_rhs("x_dot", jacp@q_dot)

        model.set_rhs("q", q_dot)
        model.set_rhs("q_dot", q_ddot)

        # model.set_rhs("u_prev", tau)  # Update previous control for the next step
        # model.set_alg('rate_constraint', ca.fabs(tau - u_prev) <= delta_u_max)

        model.setup()
        return model

    def create_mpc(self, model):
        mpc = do_mpc.controller.MPC(model)
        n_horizon = HORIZON #32
        t_step = 0.001

        setup_mpc = {
            "n_horizon": n_horizon,
            "t_step": t_step,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 3,
            "collocation_ni": 2,
            "store_full_solution": True,
            "nlpsol_opts": {'ipopt.max_iter': 10}
        }
        mpc.set_param(**setup_mpc)
        # trajectory = self.get_trajectory(self.trajectory_id)

        # target position
        target_pos = TARGET_POS   #.reshape(3, 1)
        position_error= self.compute_task_space_error(target_pos)
        # mterm = 100*ca.sumsqr(position_error)
        mterm = P[0,0]*(model.x["x"][0]-model.tvp["target_x_states"][0])**2 + P[1,1]*(model.x["x"][1]-model.tvp["target_x_states"][1])**2 + P[2,2]*(model.x["x"][2]-model.tvp["target_x_states"][2])**2
        # mterm = ca.sumsqr(model.x["x"] - model.tvp["target_x_states"])
        # mterm = (model.x["x"] - model.tvp["target_x_states"]).T@Q@(model.x["x"] - model.tvp["target_x_states"])

        # mterm = ca.sumsqr(
        #     model.x["q"] - model.tvp["target_joint_states"]
        # )  # Terminal cost
        # print(f'q --{model.x["q"]}') # joints: [q_0, q_1, q_2, q_3, q_4, q_5, q_6]
        # lterm = ca.sumsqr(model.x["x"] - model.tvp["target_x_states"]) # + 0.5*ca.sumsqr(model.u["tau"]) # 1000*cost # Stage cost
        lterm = Q[0,0]*(model.x["x"][0]-model.tvp["target_x_states"][0])**2 + Q[1,1]*(model.x["x"][1]-model.tvp["target_x_states"][1])**2 + Q[2,2]*(model.x["x"][2]-model.tvp["target_x_states"][2])**2

        # nmpc cost function
        def cost_terms (model):
            # Weights
            Q_l = np.diag(100,100,100,100,100,100,100,1,1,1,1,1,1,1) # lterm
            Q_m = np.diag(100,100,100,100,100,100,100,1,1,1,1,1,1,1) # mterm
            R = np.diag(0.01,0.01,0.01,0.01,0.01,0.01,0.01) # rterm
            
            cost = 0

            # stage cost (lterm)
            
            # nmpc_lterm 

        tvp_template = mpc.get_tvp_template()

        # def tvp_fun(t_now):
        #     traj = self.get_trajectory(trajectory_id=self.trajectory_id, t_now=t_now)
        #     for k in range(n_horizon + 1):
        #         tvp_template["_tvp", k, "target_joint_states"] = traj
        #     return tvp_template
        
        def tvp_fixed_fun(t_now):
            # Define a fixed target x position
            fixed_target = FIXED_TARGET # np.array([[0], [-0.785], [0], [-2.356], [0], [1.571], [0.785]])  # Example target positions

            # Set the same target for the whole prediction horizon
            for k in range(n_horizon + 1):
                tvp_template["_tvp", k, "target_x_states"] = fixed_target


            # # Define a fixed initial x position
            # fixed_initial = np.array([[0.088], [0], [0.926]]) # np.array([[0], [-0.785], [0], [-2.356], [0], [1.571], [0.785]])  # Example target positions

            # # Set the same target for the whole prediction horizon
            # for k in range(n_horizon + 1):
            #     tvp_template["_tvp", k, "initial_x_states"] = fixed_initial

            # # Define fixed obstacle center
            # fixed_obs_center = np.array([[0.15], [0.15], [0.75]])

            # # Set the same target for the whole prediction horizon
            # for k in range(n_horizon + 1):
            #     tvp_template["_tvp", k, "target_obs_center"] = fixed_obs_center

            return tvp_template

        mpc.set_tvp_fun(tvp_fixed_fun)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        # mpc.set_rterm(R = 1)
        mpc.set_rterm(tau=R)  # Regularization term for control inputs
        delta_u_max = 2

        # Define constraints
        mpc.bounds["lower", "_x", "q"] = -np.pi
        mpc.bounds["upper", "_x", "q"] = np.pi

        # print(f'delta -- {self.data.ctrl[:7] - delta_u_max }')

        mpc.bounds["lower", "_u", "tau"] = -5 # -3
        mpc.bounds["upper", "_u", "tau"] = 5  # self.data.ctrl[:7] + delta_u_max

        # nonlinear constraints
        # print(f'model.x["x_0"] -- {model.x["x"][0]}')
        # obs_distance_constraint = -((model.x["x"][0]-model.tvp["target_obs_center"][0])**2 + (model.x["x"][1]-model.tvp["target_obs_center"][1])**2 + (model.x["x"][2]-model.tvp["target_obs_center"][2])**2) + 0.15
        # mpc.set_nl_cons('obs_distance_constraint', obs_distance_constraint)

        mpc.setup()
        return mpc
    
    def mpc_cost(self, predicted_states, predicted_controls, Q, R, P):
        cost = 0

        # initial cost
        x_0 = predicted_states[:,0,0]
        cost = Q[0,0]*(x_0[0]-TARGET_POS[0])**2 + Q[1,1]*(x_0[1]-TARGET_POS[1])**2 + Q[2,2]*(x_0[2]-TARGET_POS[2])**2

        # stage cost
        i = 0
        for i in range(predicted_controls.shape[1]-1):
            x_i = predicted_states[:,i+1,0]
            u_i_1 = predicted_controls[:,i,0]
            u_i = predicted_controls[:,i+1,0]
            delta_u = u_i - u_i_1
            # print(0.5*(ca.sumsqr(u_i)))
            cost += Q[0,0]*(x_i[0]-TARGET_POS[0])**2 + Q[1,1]*(x_i[1]-TARGET_POS[1])**2 + Q[2,2]*(x_i[2]-TARGET_POS[2])**2 + R*(ca.sumsqr(delta_u))

        # terminal cost
        x_end = predicted_states[:,-1,0]
        # u_end = predicted_controls[:,-1,0]
        # u_end_1 = predicted_controls[:,-2,0]
        cost += P[0,0]*(x_end[0]-TARGET_POS[0])**2 + P[1,1]*(x_end[1]-TARGET_POS[1])**2 + P[2,2]*(x_end[2]-TARGET_POS[2])**2 # + 0.5*(ca.sumsqr(u_end-u_end_1))
        # print(terminal_cost)
        # terminal_cost += 0.5*(ca.sumsqr(u_end))
        # cost += terminal_cost # + 0.5*(ca.sumsqr(u_end))

        return cost

    def _simulate_mpc_mujoco(self, mpc, panda, data):
        x0 = np.zeros((20, 1))
        mpc.x0 = x0
        u0_initial_guess = U_INI_GUESS
        mpc.u0 = u0_initial_guess
        mpc.set_initial_guess()

        joint_states = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        joint_inputs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        x_states = {1: [], 2: [], 3: []}
        mpc_cost = []
        abs_distance = []

        current_step = 0

        # control update step
        mujoco_time_step = 0.001      # MuJoCo timestep
        controller_sample_time = CONTROLLER_SAMPLE_TIME  # Controller sampling time
        steps_per_control_update = int(controller_sample_time / mujoco_time_step)  # 3 steps
        max_steps = SUM_CTL_STEPS*steps_per_control_update
    
        # with mujoco.viewer.launch_passive(panda, data) as viewer:
        # start = time.time()
        # elapsed_time = time.time() - start
        while current_step < max_steps: # viewer.is_running():
            # step_start = time.time()

            # position_error, jacp = self.compute_task_space_error(target_pos)
            # joint_velocity_command = np.dot(np.linalg.pinv(jacp), position_error)
            # data.qvel[:7] = joint_velocity_command[:7].flatten()

            # mujoco.mj_step(panda, data)

            if current_step % steps_per_control_update == 0:
                # t = 0
                end_position = self.data.body("hand").xpos.copy()
                print(f'-------------------------------------------------------------------------')
                print(f'end_position -- {end_position}')
                print(f'-------------------------------------------------------------------------')
                distance = np.linalg.norm(end_position.reshape(3, 1) - TARGET_POS)
                print(f'distance -- {distance}')
                abs_distance.append(distance)

                # print(f'cost -- {(end_position[0]-TARGET_POS[0])**2 + (end_position[1]-TARGET_POS[1])**2 + (end_position[2]-TARGET_POS[2])**2}')
                # distance_cost.append((end_position[0]-TARGET_POS[0])**2 + (end_position[1]-TARGET_POS[1])**2 + (end_position[2]-TARGET_POS[2])**2)

                # distance_cost[1].append((end_position[0]-TARGET_POS[0])**2)
                # distance_cost[2].append((end_position[1]-TARGET_POS[1])**2)
                # distance_cost[3].append((end_position[2]-TARGET_POS[2])**2)


                for i in range(3):
                    x_states[i + 1].append(end_position[i])

                # Position Jacobian
                jacp, _ = self.compute_jacobian(self.panda, self.data, TARGET_POS) # 3*9
                jacp = jacp[:, :7] # 3*7

                q_current = np.array(data.qpos).reshape(-1, 1)
                q_dot_current = np.array(data.qvel).reshape(-1, 1)
                x_current = np.array(data.xpos[9,:]).reshape(-1, 1)
                x0[:7] = q_current[:7]
                x0[7:14] = q_dot_current[:7]
                x0[14:17] = x_current
                x0[17:20] = ca.mtimes(jacp, q_dot_current[:7])
                # x0[20:27] = data.ctrl[:7].reshape(-1, 1)

                for i in range(7):
                    joint_states[i + 1].append(q_current[i])

            
                u0 = mpc.make_step(x0)
                data.ctrl[:7] = u0.flatten()

                predicted_states = mpc.data.prediction(('_x', 'x'))  # Predicted states for the horizon
                predicted_controls = mpc.data.prediction(('_u', 'tau'))  # Predicted controls for the horizon
                # control_along_horizon = predicted_controls[:,0:10,:]

                applied_inputs = predicted_controls[:,-1,0].reshape(-1, 1)
                for i in range(7):
                    joint_inputs[i + 1].append(applied_inputs[i])

                # calculate mpc cost
                cost = self.mpc_cost(predicted_states, predicted_controls, Q, R, P)
                print(f'cost -- {cost}')
                cost = cost.toarray().reshape(-1)
                mpc_cost.append(cost)
                
            current_step += 1
            # elapsed_time = time.time() - start
            print(f'current_step -- {current_step}')


            # data.ctrl[:7] = control_along_horizon[:,t,:].flatten()
            # print(f'control input -- {control_along_horizon[:,t,:].flatten()}')
            mujoco.mj_step(panda, data)
            # t += 1

            # with viewer.lock():
            #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
            #         data.time % 2
            #     )

            # viewer.sync()
            # time_until_next_step = panda.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

            
            # Check if the maximum runtime has been exceeded
            # elapsed_time = time.time() - start
            # if elapsed_time >= max_runtime:
            #     print("Simulation time limit reached. Exiting...")
            #     break

        return joint_states, x_states, mpc_cost, joint_inputs, abs_distance

    def simulate(self):
        return self._simulate_mpc_mujoco(self.mpc, self.data.model, self.data)
