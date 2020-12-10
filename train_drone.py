import os
import json
import torch.optim as optim
import torch

from dataset import Dataset
from drone_loss import drone_loss_function
from evaluate_drone import QuadEvaluator
from models.resnet_like_model import Net
from environments.drone_env import construct_states
from utils.plotting import plot_loss, plot_success

EPOCH_SIZE = 1000
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 50
BATCH_SIZE = 8
NR_EVAL_ITERS = 10

net = Net(20, 4)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# TESTING
# reference_data = Dataset(construct_states, normalize=True, num_states=1000)
# trainloader = torch.utils.data.DataLoader(
#     reference_data, batch_size=20, shuffle=True, num_workers=0
# )
# for i, data in enumerate(trainloader, 0):
#     inputs, current_state = data
#     optimizer.zero_grad()
#     action = net(inputs)
#     action = torch.sigmoid(action)
#     print(action)
# print(fail)

reference_data = Dataset(
    construct_states, normalize=True, num_states=EPOCH_SIZE
)
(STD, MEAN) = (reference_data.std, reference_data.mean)

loss_list, success_mean_list, success_std_list = list(), list(), list()

for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    state_data = Dataset(
        construct_states,
        normalize=True,
        mean=MEAN,
        std=STD,
        num_states=EPOCH_SIZE
    )
    trainloader = torch.utils.data.DataLoader(
        state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    eval_env = QuadEvaluator(net, MEAN, STD)
    suc_mean, suc_std = eval_env.stabilize(nr_iters=NR_EVAL_ITERS)
    success_mean_list.append(suc_mean)
    success_std_list.append(suc_std)
    print(f"Epoch {epoch}: Time: {round(suc_mean, 1)} ({round(suc_std, 1)})")

    running_loss = 0
    try:
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, current_state = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            actions = net(inputs)
            actions = torch.sigmoid(actions)

            # compute loss + backward + optimize
            loss = drone_loss_function(current_state, actions)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % PRINT == PRINT - 1:
                print('Loss: %.3f' % (running_loss / PRINT))
                loss_list.append(running_loss / PRINT)
                running_loss = 0.0
    except KeyboardInterrupt:
        break

SAVE = os.path.join("trained_models/drone/test_model")
if not os.path.exists(SAVE):
    os.makedirs(SAVE)
# save std for normalization
param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)
#
torch.save(net, os.path.join(SAVE, "model_quad"))
plot_loss(loss_list, SAVE)
plot_success(success_mean_list, success_std_list, SAVE)
print("finished and saved.")
