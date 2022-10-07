"""
Standard MPC for Passing through a dynamic gate
"""
import casadi as ca
import numpy as np
import time
from os import system

from neural_control.dynamics.quad_dynamics_flightmare import (
    FlightmareDynamicsMPC
)
from neural_control.dynamics.quad_dynamics_simple import SimpleDynamicsMPC
from neural_control.dynamics.fixed_wing_2D import fixed_wing_dynamics_mpc
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamicsMPC
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamicsMPC

#
class MPC(object):
    """
    Nonlinear MPC
    """

    def __init__(self, horizon=20, dt=0.05, dynamics="high_mpc", **kwargs):
        """
        Nonlinear MPC for quadrotor control        
        """

        self.dynamics_model = dynamics

        # Time constant
        self._T = horizon * dt
        self._dt = dt
        self._N = horizon

        # Gravity
        self._gz = 9.81

        # add to reference
        self.addon = np.swapaxes(
            np.vstack(
                (
                    np.expand_dims(np.arange(0, self._T - 0.001, self._dt), 0),
                    np.zeros((1, self._N)), np.zeros((1, self._N)) + 10
                )
            ), 1, 0
        )

        # cost matrix for tracking the pendulum motion
        if self.dynamics_model == "high_mpc":
            self._initParamsHighMPC()
        elif self.dynamics_model in ["simple_quad", "flightmare"]:
            self._initParamsSimpleQuad()
        elif self.dynamics_model == "fixed_wing_2D":
            self._initParamsFixedWing_2D()
        elif self.dynamics_model == "fixed_wing_3D":
            self._initParamsFixedWing_3D()
        elif self.dynamics_model == "cartpole":
            self._initParamsCartpole()

        # cost matrix for tracking the goal point
        self._Q_goal = np.zeros((self._s_dim, self._s_dim))

        self._initDynamics()

    def _initParamsHighMPC(self):
        # Quadrotor constant
        self._w_max_xy = 6.0
        self._w_min_xy = -6.0
        self._thrust_min = 2.0
        self._thrust_max = 20.0
        # state dimension (px, py, pz,           # quadrotor position
        #                  qw, qx, qy, qz,       # quadrotor quaternion
        #                  vx, vy, vz,           # quadrotor linear velocity
        self._s_dim = 10
        # action dimensions (c_thrust, wx, wy, wz)
        self._u_dim = 4

        # cost matrix for the action
        self._Q_u = np.diag([0.1 for _ in range(self._u_dim)])
        self._Q_pen = np.diag([0, 100, 100, 0, 0, 0, 0, 0, 10, 10])
        # initial state and control action
        self._quad_s0 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._quad_u0 = [9.81, 0.0, 0.0, 0.0]
        # default u
        self._default_u = [self._gz, 0, 0, 0]

     def _initParamsCartpole(self):
        self._s_dim = 15
        self._u_dim = 1
        self._w_min_xy = -1
        self._w_max_xy = 1
        self._thrust_min = -1
        self._thrust_max = 1
        self._Q_u = np.diag([0])
        self._Q_pen = np.diag([0, 3, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # initial states
        self._quad_s0 = np.zeros(15).tolist()
        self._quad_u0 = [0]
        # default u
        self._default_u = [0]

    def _initParamsSimpleQuad(self):
        # Quadrotor constant
        self._w_max_xy = 1
        self._w_min_xy = 0
        self._thrust_min = 0
        self._thrust_max = 1
        self._s_dim = 12
        self._u_dim = 4
        # cost matrix for the action
        self._Q_u = np.diag([50, 1, 1, 1])
        self._Q_pen = np.diag([100, 100, 100, 0, 0, 0, 10, 10, 10, 1, 1, 1])
        # initial state and control action TODO
        self._quad_s0 = (np.zeros(12)).tolist()
        self._quad_u0 = [.5, .5, .5, .5]
        # default u
        self._default_u = [.5, .5, .5, .5]

    def _initParamsFixedWing_2D(self):
        self._s_dim = 6
        self._u_dim = 2
        self._w_min_xy = 0
        self._w_max_xy = 1
        self._thrust_min = 0
        self._thrust_max = 1
        # cost matrix for the action
        self._Q_u = np.diag([0, 10])
        self._Q_pen = np.diag([1000, 1000, 0, 0, 0, 0])
        # initial states
        self._quad_s0 = np.array([0, 0, 10, 0, 0, 0]).tolist()
        self._quad_u0 = [.25, .5]
        # default u
        self._default_u = [.25, .5]

    def _initParamsFixedWing_3D(self):
        self._s_dim = 12
        self._u_dim = 4
        self._w_min_xy = 0
        self._w_max_xy = 1
        self._thrust_min = 0
        self._thrust_max = 1
        self._Q_u = np.diag([0, 10, 10, 10])
        self._Q_pen = np.diag([1000, 1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # initial states
        self._quad_s0 = np.zeros(12).tolist()
        self._quad_s0[3] = 11
        self._quad_u0 = [.25, .5, .5, .5]
        # default u
        self._default_u = [.25, .5, .5, .5]

    def _initDynamics(self, ):
        # # # # # # # # # # # # # # # # # # #
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # #

        # # Fold
        if self.dynamics_model == "high_mpc":
            F = self.drone_dynamics_high_mpc(self._dt)
        elif self.dynamics_model == "simple_quad":
            dyn = SimpleDynamicsMPC()
            F = dyn.drone_dynamics_simple(self._dt)
        elif self.dynamics_model == "flightmare":
            dyn = FlightmareDynamicsMPC()
            F = dyn.drone_dynamics_flightmare(self._dt)
        elif self.dynamics_model == "fixed_wing_2D":
            F = fixed_wing_dynamics_mpc(self._dt)
        elif self.dynamics_model == "fixed_wing_3D":
            dyn = FixedWingDynamicsMPC()
            F = dyn.simulate_fixed_wing(self._dt)
        elif self.dynamics_model == "cartpole":
            dyn = CartpoleDynamicsMPC(self.modified_dynamics)
            F = dyn.simulate_cartpole(self._dt)
        fMap = F.map(self._N, "openmp")  # parallel

        # # # # # # # # # # # # # # #
        # ---- loss function --------
        # # # # # # # # # # # # # # #

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)
        Delta_p = ca.SX.sym("Delta_p", self._s_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)

        #
        cost_goal = Delta_s.T @ self._Q_goal @ Delta_s
        cost_gap = Delta_p.T @ self._Q_pen @ Delta_p
        cost_u = Delta_u.T @ self._Q_u @ Delta_u

        #
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_gap = ca.Function('cost_gap', [Delta_p], [cost_gap])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        #
        # # # # # # # # # # # # # # # # # # # #
        # # ---- Non-linear Optimization -----
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constrait functions, lbg < g
        self.ubg = []  # upper bound of constrait functions, g < ubg

        u_min = [self._thrust_min
                 ] + [self._w_min_xy for _ in range(self._u_dim - 1)]
        u_max = [self._thrust_max
                 ] + [self._w_max_xy for _ in range(self._u_dim - 1)]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._s_dim)]
        x_max = [+x_bound for _ in range(self._s_dim)]
        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        P = ca.SX.sym(
            "P", self._s_dim + (self._s_dim + 3) * self._N + self._s_dim
        )
        X = ca.SX.sym("X", self._s_dim, self._N + 1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        #
        X_next = fMap(X[:, :self._N], U)

        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._quad_s0
        self.lbw += x_min
        self.ubw += x_max

        # # starting point.
        self.nlp_g += [X[:, 0] - P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._quad_u0
            self.lbw += u_min
            self.ubw += u_max

            # retrieve time constant
            # idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            # idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)
            # time_k = P[ idx_k : idx_k_end]

            # cost for tracking the goal position
            cost_goal_k, cost_gap_k = 0, 0
            if k >= self._N - 1:  # The goal postion.
                delta_s_k = (
                    X[:, k + 1] - P[self._s_dim + (self._s_dim + 3) * self._N:]
                )
                cost_goal_k = f_cost_goal(delta_s_k)
            else:
                # cost for tracking the moving gap
                delta_p_k = (X[:, k+1] - P[self._s_dim+(self._s_dim+3)*k : \
                    self._s_dim+(self._s_dim+3)*(k+1)-3])
                cost_gap_k = f_cost_gap(delta_p_k)

            delta_u_k = U[:, k] - self._default_u

            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k + cost_gap_k

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k + 1]]
            self.nlp_w0 += self._quad_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {
            'f': self.mpc_obj,
            'x': ca.vertcat(*self.nlp_w),
            'p': P,
            'g': ca.vertcat(*self.nlp_g)
        }

        # # # # # # # # # # # # # # # # # # #
        # -- ipopt
        # # # # # # # # # # # # # # # # # # #
        ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

    def solve(self, ref_states):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        if self.dynamics_model == "high_mpc":
            start = ref_states[:self._s_dim]
            end = ref_states[-self._s_dim:]
            end[3:7] = [0 for _ in range(4)]
            middle_ref_states = np.array(ref_states[self._s_dim:-self._s_dim]
                                         ).reshape((10, 13))
            middle_ref_states[:, 3:7] = 0
            ref_states = start + middle_ref_states.flatten().tolist() + end

        # print(
        #     len(self.nlp_w0), len(self.lbw), len(self.ubw), len(ref_states),
        #     len(self.lbg), len(self.ubg)
        # )
        # tic = time.time()
        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            p=ref_states,
            lbg=self.lbg,
            ubg=self.ubg
        )
        # print(time.time() - tic)
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim:self._s_dim + self._u_dim]

        # Warm initialization
        self.nlp_w0 = list(
            sol_x0[self._s_dim + self._u_dim:2 * (self._s_dim + self._u_dim)]
        ) + list(sol_x0[self._s_dim + self._u_dim:])
        # print(self.nlp_w0)
        # print(len(self.nlp_w0))
        # #
        x0_array = np.reshape(
            sol_x0[:-self._s_dim], newshape=(-1, self._s_dim + self._u_dim)
        )
        # np.set_printoptions(suppress=1, precision=3)
        # print(x0_array[1:, :9])
        return opt_u, x0_array

    def preprocess_quad(self, current_state, ref_states):
        """
        current_state: list / array of len 12
        ref_states: array of shape (horizon, 9) with pos, vel, acc
        """
        # [0 for _ in range(len(current_state))]
        # modify the reference traj to input it into mpc
        changed_middle_ref_states = np.zeros((self._N, len(current_state)))
        changed_middle_ref_states[:, :3] = ref_states[:, :3]
        changed_middle_ref_states[:, 3:6] = ref_states[:, 3:6]
        changed_middle_ref_states[:, 6:9] = ref_states[:, 6:9]

        # no goal point for now
        # goal_state = changed_middle_ref_states[-1].copy().tolist()
        goal_state = np.zeros(self._s_dim)
        goal_state[:3] = (
            2 * changed_middle_ref_states[-1, :3] -
            changed_middle_ref_states[-2, :3]
        )
        goal_state[6:9] = changed_middle_ref_states[-1, 6:9]

        # print("-------------------")
        # np.set_printoptions(suppress=True, precision=3)
        # print(changed_middle_ref_states[:-1, :3])

        # apped three mysterious entries:
        high_mpc_reference = np.hstack((changed_middle_ref_states, self.addon))

        flattened_ref = (
            current_state.tolist() + high_mpc_reference.flatten().tolist() +
            goal_state.tolist()
        )
        return flattened_ref

    def preprocess_fixed_wing(self, current_state, ref_states):
        """
        Construct a reference as required by MPC from the current state and
        the desired ref
        """
        vec_to_target = ref_states - current_state[:3]
        vec_norm = np.linalg.norm(vec_to_target)
        speed = np.sqrt(np.sum(current_state[3:6]**2))
        vec_len_per_step = speed * self._dt
        vector_per_step = vec_to_target * (vec_len_per_step / vec_norm)

        middle_ref_states = np.zeros((self._N + 1, len(current_state)))
        for i in range(self._N + 1):
            middle_ref_states[
                i, :3] = current_state[:3] + (i + 1) * vector_per_step

        # np.set_printoptions(suppress=True, precision=3)
        # # print(opt_u.tolist())
        # print("middle_ref_states")
        # print(np.vstack((current_state, middle_ref_states)))

        # goal point is last point of middle ref
        goal_state = middle_ref_states[-1]

        high_mpc_reference = np.hstack((middle_ref_states[:-1], self.addon))

        flattened_ref = (
            current_state.tolist() + high_mpc_reference.flatten().tolist() +
            goal_state.tolist()
        )

        return flattened_ref

    def preprocess_cartpole(self, current_state):
        # interpolate theta
        inter_theta = np.linspace(current_state[2], 0, self._N + 2)
        inter_theta_dot = np.linspace(current_state[3], 0, self._N + 2)
        inter_x = np.linspace(current_state[0], 0, self._N + 2)
        inter_x_dot = np.linspace(current_state[1], 0, self._N + 2)

        middle_ref_states = np.zeros((self._N + 2, len(current_state)))
        middle_ref_states[:, 0] = inter_x
        middle_ref_states[:, 1] = inter_x_dot
        middle_ref_states[:, 2] = inter_theta
        middle_ref_states[:, 3] = inter_theta_dot

        goal_state = middle_ref_states[-1]

        high_mpc_reference = np.hstack((middle_ref_states[1:-1], self.addon))
        # print(current_state)
        # print(high_mpc_reference)
        # print()

        flattened_ref = (
            current_state.tolist() + high_mpc_reference.flatten().tolist() +
            goal_state.tolist()
        )
        return flattened_ref
        
    def predict_actions(self, current_state, ref_states):
        if self.dynamics_model in ["simple_quad", "flightmare"]:
            preprocessed_ref = self.preprocess_quad(current_state, ref_states)
        elif self.dynamics_model in ["fixed_wing_2D", "fixed_wing_3D"]:
            preprocessed_ref = self.preprocess_fixed_wing(
                current_state, ref_states
            )
        elif self.dynamics_model == "cartpole":
            preprocessed_ref = self.preprocess_cartpole(ref_states[0].numpy())
        action, _ = self.solve(preprocessed_ref)
        return np.array([action[:, 0]])

    def drone_dynamics_high_mpc(self, dt):

        self.f = self.get_dynamics_high_mpc()

        M = 4  # refinement
        DT = dt / M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            k1 = DT * self.f(X, U)
            k2 = DT * self.f(X + 0.5 * k1, U)
            k3 = DT * self.f(X + 0.5 * k2, U)
            k4 = DT * self.f(X + k3, U)
            #
            X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # Fold
        F = ca.Function('F', [X0, U], [X])
        return F

    def get_dynamics_high_mpc(self):
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        #
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), \
            ca.SX.sym('qz')
        #
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')

        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), \
            ca.SX.sym('wy'), ca.SX.sym('wz')

        # -- conctenated vector
        self._u = ca.vertcat(thrust, wx, wy, wz)

        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #

        x_dot = ca.vertcat(
            vx, vy, vz, 0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy),
            2 * (qw * qy + qx * qz) * thrust, 2 * (qy * qz - qw * qx) * thrust,
            (qw * qw - qx * qx - qy * qy + qz * qz) * thrust - self._gz
            # (1 - 2*qx*qx - 2*qy*qy) * thrust - self._gz
        )

        #
        func = ca.Function(
            'f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode']
        )
        return func
