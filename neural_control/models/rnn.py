import torch
import torch.nn as nn
import torch.functional as F

# from neural_control.environments.drone_dynamics import simulate_quadrotor


class LSTM_NEW(nn.Module):

    def __init__(
        self, state_dim, horizon, ref_dim, nr_actions_predict, conv=True
    ):
        print("Using LSTM cell")
        super(LSTM_NEW, self).__init__()
        self.state_dim = state_dim
        self.ref_dim = ref_dim

        # normal logic for processing the reference trajectory
        self.conv_ref = nn.Conv1d(ref_dim, 20, kernel_size=3)
        self.horizon = horizon
        self.conv = conv
        self.reshape_len = 20 * (horizon - 2) if conv else 64
        self.ref_in = nn.Linear(horizon * ref_dim, 64)
        self.fc_out = nn.Linear(64, nr_actions_predict)

        # init lstm cell
        self.lstm = nn.LSTMCell(state_dim + self.reshape_len, 64)
        self.reset_hidden_state(1)

    def reset_hidden_state(self, batch_size=1):
        # we need to reset the hidden state whenever starting a new sequence (?)
        self.hidden_state = torch.randn(batch_size, 64)
        self.cell_state = torch.randn(batch_size, 64)

    def forward(self, state, ref):
        # process state and reference differently
        if self.conv:
            # ref = torch.reshape(ref, (-1, self.ref_dim, 3))
            ref = torch.transpose(ref, 1, 2)
            ref = torch.relu(self.conv_ref(ref))
            ref = torch.reshape(ref, (-1, self.reshape_len))
        else:
            ref = torch.tanh(self.ref_in(ref))
        # concatenate
        x = torch.hstack((state, ref))
        # pass through LSTM
        self.hidden_state, self.cell_state = self.lstm(
            x, (self.hidden_state, self.cell_state)
        )
        # Output layer: The current hidden state is transformed into an output
        return self.fc_out(self.hidden_state)
