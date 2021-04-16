import numpy as np
import torch
import contextlib
import time
import torch.optim as optim


@contextlib.contextmanager
def dummy_context():
    yield None


class NetworkWrapper:

    def __init__(
        self,
        model,
        dataset,
        optimizer=None,
        horizon=5,
        max_drone_dist=0.1,
        render=0,
        dt=0.02,
        take_every_x=1000,
        **kwargs
    ):
        self.dataset = dataset
        self.net = model
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.training_means = None
        self.render = render
        self.dt = dt
        self.optimizer = optimizer
        self.take_every_x = take_every_x
        self.action_counter = 0

        # four control signals
        self.action_dim = 4

    def predict_actions(self, current_np_state, ref_states):
        """
        Predict an action for the current state. This function is used by all
        evaluation functions
        """
        # determine whether we also add the sample to our train data
        add_to_dataset = (self.action_counter + 1) % self.take_every_x == 0
        # preprocess state
        in_state, current_state, ref, _ = self.dataset.get_and_add_eval_data(
            current_np_state.copy(), ref_states, add_to_dataset=add_to_dataset
        )

        with torch.no_grad():
            suggested_action = self.net(in_state, ref)

            suggested_action = torch.sigmoid(suggested_action)

            suggested_action = torch.reshape(
                suggested_action, (1, self.horizon, self.action_dim)
            )

        numpy_action_seq = suggested_action[0].detach().numpy()
        # print([round(a, 2) for a in numpy_action_seq[0]])
        # keep track of actions
        self.action_counter += 1
        return numpy_action_seq


class FixedWingNetWrapper:

    def __init__(self, model, dataset, horizon=1, take_every_x=1000, **kwargs):
        self.net = model
        self.dataset = dataset
        self.horizon = horizon
        self.action_dim = 4
        self.action_counter = 0
        self.take_every_x = take_every_x

    def predict_actions(self, state, ref_state):
        # determine whether we also add the sample to our train data
        add_to_dataset = (self.action_counter + 1) % self.take_every_x == 0

        normed_state, _, normed_ref, _ = self.dataset.get_and_add_eval_data(
            state, ref_state, add_to_dataset=add_to_dataset
        )
        with torch.no_grad():
            suggested_action = self.net(normed_state, normed_ref)
            suggested_action = torch.sigmoid(suggested_action)[0]

            suggested_action = torch.reshape(
                suggested_action, (self.horizon, self.action_dim)
            )

        self.action_counter += 1
        return suggested_action.detach().numpy()
