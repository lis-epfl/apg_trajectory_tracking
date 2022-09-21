import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_SIZE = 10  # one action variable between -1 and 1
DIM = 4  # input dimension


class Net(nn.Module):

    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc0 = nn.Linear(in_size, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, out_size)

    def forward(self, x):
        x[:, 0] *= 0
        x = torch.tanh(self.fc0(x))
        # x = x * torch.from_numpy(np.array([0, 1, 1, 1]))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc_out(x))
        return x


class StateToImg(nn.Module):

    def __init__(self, width=100, height=120):
        super(StateToImg, self).__init__()
        self.img_height = height
        self.img_width = width
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc_out = nn.Linear(256, width * height)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc_out(x))
        x = torch.reshape(x, (-1, self.img_width, self.img_height))
        return x


class ImageControllerNet(nn.Module):

    def __init__(self, img_height, img_width, out_size=1, nr_img=5):
        super(ImageControllerNet, self).__init__()
        # all raw images and the subtraction
        self.conv1 = nn.Conv2d(nr_img * 2 - 1, 10, 5)
        self.conv2 = nn.Conv2d(10, 2, 3)

        self.flat_img_size = 2 * (img_height - 6) * (img_width - 6)

        self.fc1 = nn.Linear(self.flat_img_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, out_size)

    def forward(self, image):
        cat_all = [image]
        for i in range(image.size()[1] - 1):
            cat_all.append(
                torch.unsqueeze(image[:, i + 1] - image[:, i], dim=1)
            )
        sub_images = torch.cat(cat_all, dim=1)
        conv1 = torch.relu(self.conv1(sub_images.float()))
        conv2 = torch.relu(self.conv2(conv1))

        x = conv2.reshape((-1, self.flat_img_size))

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc_out(x))
        return x


HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5  # original = 5
STRIDE = 2  # original = 2


class ImageControllerNetDQN(nn.Module):

    def __init__(self, h, w, out_size=1, nr_img=3):
        super(ImageControllerNetDQN, self).__init__()
        self.conv1 = nn.Conv2d(
            nr_img, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE
        )
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(
            HIDDEN_LAYER_1,
            HIDDEN_LAYER_2,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE
        )
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(
            HIDDEN_LAYER_2,
            HIDDEN_LAYER_3,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE
        )
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, out_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))