import numpy as np
from neural_control.baselines.mpc import MPC

from evaluate_drone import QuadEvaluator


class MPCWrapper(QuadEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = self.dt
        self.T = self.dt * self.horizon
        self.mpc = MPC(self.T, self.dt, dynamics="simple_quad")

    def predict_actions(self, current_state, ref_states):
        """
        current_state: list / array of len 12
        ref_states: array of shape (horizon, 9) with pos, vel, acc
        """
        # no goal point for now
        goal_state = [0 for _ in range(len(current_state))]
        # modify the reference traj to input it into mpc
        changed_middle_ref_states = np.zeros(
            (self.horizon, len(current_state))
        )
        changed_middle_ref_states[:, :3] = ref_states[:, :3]
        changed_middle_ref_states[:, 6:9] = ref_states[:, 3:6]
        # apped three mysterious entries:
        addon = np.swapaxes(
            np.vstack(
                (
                    np.expand_dims(np.arange(0, self.T, self.dt),
                                   0), np.zeros((1, self.horizon)),
                    np.zeros((1, self.horizon)) + 10
                )
            ), 1, 0
        )
        high_mpc_reference = np.hstack((changed_middle_ref_states, addon))
        print("should be (10,12 + 3):", high_mpc_reference.shape)

        flattened_ref = (
            current_state.tolist() + high_mpc_reference.flatten().tolist() +
            goal_state
        )
        print(len(flattened_ref), "should be 12 + 150 + 12")

        action, predicted_states = self.mpc.solve(flattened_ref)
        print(action)
        return action
