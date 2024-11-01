import numpy as np
import casadi as ca
import do_mpc
import mujoco
import mujoco.viewer
import time

TARGET_JOINT = np.array([[0], [-0.785], [0], [-2.356], [0], [1.571], [0.785]]).reshape(7, 1)
TARGET_POS = np.array([0.1, 0.3, 0.45]).reshape(3, 1)
Q = np.diag([1,1,1])
R = np.diag([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
P = np.diag([1,1,1])


class Joint_MPC:

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

        # Define the time variable trajectory
        target_joint_states = model.set_variable(
            var_type="_tvp", var_name="target_joint_states", shape=(7, 1)
        )

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

        model.setup()
        return model

    def create_mpc(self, model):
        mpc = do_mpc.controller.MPC(model)
        n_horizon = 10 #32
        t_step = 0.05

        setup_mpc = {
            "n_horizon": n_horizon,
            "t_step": 0.05,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 3,
            "collocation_ni": 2,
            "store_full_solution": True,
        }
        mpc.set_param(**setup_mpc)
        # trajectory = self.get_trajectory(self.trajectory_id)

        # target position
        # target_pos = TARGET_POS   #.reshape(3, 1)
        # position_error= self.compute_task_space_error(target_pos)
        # mterm = 100*ca.sumsqr(position_error)
        # cost = (model.x["x"][0]-model.tvp["target_x_states"][0])**2 + (model.x["x"][1]-model.tvp["target_x_states"][1])**2 + (model.x["x"][2]-model.tvp["target_x_states"][2])**2
        # mterm = 1000*cost
        mterm = ca.sumsqr(model.x["q"] - model.tvp["target_joint_states"])

        # mterm = ca.sumsqr(
        #     model.x["q"] - model.tvp["target_joint_states"]
        # )  # Terminal cost
        # print(f'q --{model.x["q"]}') # joints: [q_0, q_1, q_2, q_3, q_4, q_5, q_6]
        lterm = ca.sumsqr(model.x["q"] - model.tvp["target_joint_states"]) # + 0.5*ca.sumsqr(model.u["tau"]) # 1000*cost # Stage cost

        tvp_template = mpc.get_tvp_template()

        
        def tvp_fixed_fun(t_now):
            # Define a fixed target x position
            fixed_target = np.array([[0], [-0.785], [0], [-2.356], [0], [1.571], [0.785]]) # np.array([[0], [-0.785], [0], [-2.356], [0], [1.571], [0.785]])  # Example target positions

            # Set the same target for the whole prediction horizon
            for k in range(n_horizon + 1):
                tvp_template["_tvp", k, "target_joint_states"] = fixed_target

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
        mpc.set_rterm(tau=0.02)  # Regularization term for control inputs

        # Define constraints
        mpc.bounds["lower", "_x", "q"] = -np.pi
        mpc.bounds["upper", "_x", "q"] = np.pi

        # mpc.bounds["lower", "_u", "tau"] = -1000
        # mpc.bounds["upper", "_u", "tau"] = 1000

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
        cost = (x_0[0]-TARGET_POS[0])**2 + (x_0[1]-TARGET_POS[1])**2 + (x_0[2]-TARGET_POS[2])**2

        # stage cost
        for i in range(predicted_controls.shape[1]-1):
            x_i = predicted_states[:,i+1,0]
            u_i = predicted_controls[:,i,0]
            # print(0.5*(ca.sumsqr(u_i)))
            cost += (x_i[0]-TARGET_POS[0])**2 + (x_i[1]-TARGET_POS[1])**2 + (x_i[2]-TARGET_POS[2])**2 + (ca.sumsqr(u_i))

        # terminal cost
        x_end = predicted_states[:,-1,0]
        u_end = predicted_controls[:,-1,0]
        terminal_cost = (x_end[0]-TARGET_POS[0])**2 + (x_end[1]-TARGET_POS[1])**2 + (x_end[2]-TARGET_POS[2])**2
        print(terminal_cost)
        terminal_cost += (ca.sumsqr(u_end))
        cost += terminal_cost # + 0.5*(ca.sumsqr(u_end))

        return cost
    
    def joint_cost(self, predicted_joints, predicted_controls, Q, R, P):
        cost = 0

        # initial cost
        q_0 = predicted_joints[:,0,0]
        cost = (q_0[0]-TARGET_JOINT[0])**2 + (q_0[1]-TARGET_JOINT[1])**2 + (q_0[2]-TARGET_JOINT[2])**2 + (q_0[3]-TARGET_JOINT[3])**2 + (q_0[4]-TARGET_JOINT[4])**2 + (q_0[5]-TARGET_JOINT[5])**2 + (q_0[6]-TARGET_JOINT[6])**2

        # stage cost
        for i in range(predicted_controls.shape[1]-1):
            q_i = predicted_joints[:,i+1,0]
            u_i = predicted_controls[:,i,0]
            # print(0.5*(ca.sumsqr(u_i)))
            cost += (q_i[0]-TARGET_JOINT[0])**2 + (q_i[1]-TARGET_JOINT[1])**2 + (q_i[2]-TARGET_JOINT[2])**2 + (q_i[3]-TARGET_JOINT[3])**2 + (q_i[4]-TARGET_JOINT[4])**2 + (q_i[5]-TARGET_JOINT[5])**2 + (q_i[6]-TARGET_JOINT[6])**2 + 0.02*(ca.sumsqr(u_i))

        # terminal cost
        q_end = predicted_joints[:,-1,0]
        u_end = predicted_controls[:,-1,0]
        terminal_cost = (q_end[0]-TARGET_JOINT[0])**2 + (q_end[1]-TARGET_JOINT[1])**2 + (q_end[2]-TARGET_JOINT[2])**2 + (q_end[3]-TARGET_JOINT[3])**2 + (q_end[4]-TARGET_JOINT[4])**2 + (q_end[5]-TARGET_JOINT[5])**2 + (q_end[6]-TARGET_JOINT[6])**2
        print(terminal_cost)
        terminal_cost += 0.02*(ca.sumsqr(u_end))
        cost += terminal_cost # + 0.5*(ca.sumsqr(u_end))

        return cost

    def _simulate_mpc_mujoco(self, mpc, panda, data):
        x0 = np.zeros((20, 1))
        mpc.x0 = x0
        u0_initial_guess = ca.DM([-50,10,-50,10,-50,10,-50])
        mpc.u0 = u0_initial_guess
        mpc.set_initial_guess()

        joint_states = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        x_states = {1: [], 2: [], 3: []}
        mpc_cost = []

        max_runtime = 300

        with mujoco.viewer.launch_passive(panda, data) as viewer:
            start = time.time()
            while viewer.is_running():
                step_start = time.time()

                # position_error, jacp = self.compute_task_space_error(target_pos)
                # joint_velocity_command = np.dot(np.linalg.pinv(jacp), position_error)
                # data.qvel[:7] = joint_velocity_command[:7].flatten()
                 
                mujoco.mj_step(panda, data)

                end_position = self.data.body("hand").xpos.copy()
                print(f'-------------------------------------------------------------------------')
                print(f'end_position -- {end_position}')
                print(f'-------------------------------------------------------------------------')

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
                # x0[:7] = q_current[:7]
                # x0[7:] = q_dot_current[:7]

                for i in range(7):
                    joint_states[i + 1].append(q_current[i])

                u0 = mpc.make_step(x0)
                data.ctrl[:7] = u0.flatten()

                # Get the objective value from the solver after the optimization step
                # mpc_opt = mpc.get_tvp_template()  # Get current parameters
                # solver_stats = mpc.nlp  # Get CasADi solver statistics
                # objective_value = solver_stats['f']  # This gives the final objective (cost) value
                # print("Objective Value:", objective_value)
                # mpc_cost.append(objective_value)

                predicted_joints = mpc.data.prediction(('_x', 'q'))  # Predicted states for the horizon
                predicted_controls = mpc.data.prediction(('_u', 'tau'))  # Predicted controls for the horizon
                # print(f'range(predicted_controls.shape[1]-1) -- {range(predicted_controls.shape[1]-1)}')

                # calculate mpc cost
                cost = self.joint_cost(predicted_joints, predicted_controls, Q, R, P)
                cost = cost.toarray().reshape(-1)
                mpc_cost.append(cost)


                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        data.time % 2
                    )

                viewer.sync()
                time_until_next_step = panda.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                
                # Check if the maximum runtime has been exceeded
                elapsed_time = time.time() - start
                if elapsed_time >= max_runtime:
                    print("Simulation time limit reached. Exiting...")
                    break

        return joint_states, x_states, mpc_cost

    def simulate(self):
        return self._simulate_mpc_mujoco(self.mpc, self.data.model, self.data)