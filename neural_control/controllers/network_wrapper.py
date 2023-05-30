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
        horizon=10,
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
        self.horizon = horizon

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
            suggested_action = self.net(in_state, ref[:, :self.horizon])

            suggested_action = torch.sigmoid(suggested_action)

            if suggested_action.size()[-1] > self.action_dim:
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

            if suggested_action.size()[-1] > self.action_dim:
                suggested_action = torch.reshape(
                    suggested_action, (self.horizon, self.action_dim)
                )

        self.action_counter += 1
        return suggested_action.detach().numpy()


class CartpoleWrapper:

    def __init__(self, model, horizon=10, action_dim=1, **kwargs):
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = model

    def raw_states_to_torch(
        self, states, normalize=False, std=None, mean=None, return_std=False
    ):
        """
        Helper function to convert numpy state array to normalized tensors
        Argument states:
                One state (list of length 4) or array with x states (x times 4)
        """
        # either input one state at a time (evaluation) or an array
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)

        # save mean and std column wise
        if normalize:
            # can't use mean!
            if std is None:
                std = np.std(states, axis=0)
            if mean is None:
                mean = np.mean(states, axis=0)
            states = (states - mean) / std
            # assert np.all(np.isclose(np.std(states, axis=0), 1))
        else:
            std = 1

        # np.save("data_backup/quad_data.npy", states)

        states_to_torch = torch.from_numpy(states).float()

        # if we computed mean and std here, return it
        if return_std:
            return states_to_torch, mean, std
        return states_to_torch

    def predict_actions(self, state, ref_state):
        torch_state = self.raw_states_to_torch(state)
        action_seq = self.net(torch_state)
        if action_seq.size()[-1] > self.action_dim:
            action_seq = torch.reshape(
                action_seq, (-1, self.horizon, self.action_dim)
            )
        return action_seq


class CartpoleImageWrapper:

    def __init__(
        self,
        net,
        dataset,
        horizon=3,
        action_dim=1,
        self_play=1,
        take_every_x=5,
        **kwargs
    ):
        self.dataset = dataset
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = net
        self.self_play = (self_play == "all" or self_play > 0)
        self.action_counter = 0
        self.take_every_x = take_every_x

    def to_torch(self, inp):
        """
        To torch tensor
        """
        return torch.from_numpy(np.expand_dims(inp, 0)).float()

    def predict_actions(self, img_input, state):
        # img_input = self.to_torch(image)
        action_seq = self.net(img_input)
        action_seq = torch.reshape(
            action_seq, (-1, self.horizon, self.action_dim)
        )
        if self.self_play and (
            self.action_counter + 1
        ) % self.take_every_x == 0:
            torch_state = self.to_torch(state)
            self.dataset.add_data(img_input, torch_state, action_seq[:, 0])

        self.action_counter += 1
        return action_seq


class SequenceCartpoleWrapper(CartpoleImageWrapper):

    def predict_actions(self, state_buffer, action_buffer, network_input):

        action_seq = self.net(network_input)
        action_seq = torch.reshape(
            action_seq, (-1, self.horizon, self.action_dim)
        )
        if self.self_play and (
            self.action_counter + 1
        ) % self.take_every_x == 0:
            self.dataset.add_data(
                state_buffer, action_buffer, action_seq[:, 0]
            )
        self.action_counter += 1
        return action_seq