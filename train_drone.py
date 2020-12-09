import os
import json
import torch.optim as optim
import torch

from dataset import Dataset
from drone_loss import drone_loss_function
from evaluate_drone import QuadEvaluator
from models.resnet_like_model import Net
from environments.drone_env import construct_states
from utils.plotting import plot_loss

EPOCH_SIZE = 10000
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 10
BATCH_SIZE = 8
NR_EVAL_ITERS = 10

net = Net(20, 4)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

reference_data = Dataset(
    construct_states, normalize=True, num_states=EPOCH_SIZE
)
STD = reference_data.std

loss_list = list()

for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    state_data = Dataset(
        construct_states, normalize=True, std=STD, num_states=EPOCH_SIZE
    )
    trainloader = torch.utils.data.DataLoader(
        state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    eval_env = QuadEvaluator(net, STD)
    suc_mean, suc_std = eval_env.stabilize(nr_iters=NR_EVAL_ITERS)
    print(f"Epoch {epoch}: Time: {round(suc_mean, 1)} ({round(suc_std, 1)})")

    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, current_state = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        actions = net(inputs)
        actions = torch.sigmoid(actions)
        lam = epoch / NR_EPOCHS
        loss = drone_loss_function(current_state, actions)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % PRINT == PRINT - 1:
            print('Loss: %.3f' % (running_loss / PRINT))
            loss_list.append(running_loss / PRINT)
            running_loss = 0.0

SAVE = os.path.join("trained_models/drone/test_model")
if not os.path.exists(SAVE):
    os.makedirs(SAVE)
with open(os.path.join(SAVE, "std.json"), "w") as outfile:
    json.dump({"std": STD.tolist()}, outfile)
plot_loss(loss_list, SAVE)
torch.save(net, os.path.join(SAVE, "model_quad"))
