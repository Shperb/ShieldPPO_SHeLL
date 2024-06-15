import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import constants
from enum import Enum

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")

kernel_size = 5
stride = 1
feature_dim = 128


class ObservationType(Enum):
    Camera = 1
    Kinematics = 2
    OccupancyGrid = 3


class CameraEncoder(nn.Module):
    def __init__(self, width, height, channels):
        super(CameraEncoder, self).__init__()
        out_channels = 5
        last_out_channels = 8
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size, stride).to(device)
        self.bn1 = nn.BatchNorm2d(out_channels).to(device)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(out_channels, last_out_channels, kernel_size, stride).to(device)
        self.bn2 = nn.BatchNorm2d(last_out_channels).to(device)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(last_out_channels, last_out_channels, kernel_size, stride).to(device)
        self.bn3 = nn.BatchNorm2d(last_out_channels).to(device)
        self.w, self.h = width, height
        self.channels = channels

        def conv2d_size_out(size, pool=2):
            conv_size = (size - (kernel_size - 1) - 1) // stride + 1
            return conv_size // pool

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)), pool=1)
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)), pool=1)
        linear_input_size = conv_w * conv_h * last_out_channels  # output size of the convolutional layers
        self.output = nn.Linear(linear_input_size, feature_dim).to(device)
        self.norm = L2Norm()

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.norm(self.output(x))

    def encode(self, states):
        if len(states.shape) > 1:
            batch_size = states.size(0)
            states = states.view(batch_size, self.channels, self.w, self.h)
            encoded_states = self.forward(states)
            encoded_states = encoded_states.view(batch_size, feature_dim)
        else:
            batch_size = 1
            states = states.view(batch_size, self.channels, self.w, self.h)
            encoded_states = self.forward(states)
            encoded_states = encoded_states.view(-1)
        # encoded_states = self.forward(states.view(batch_size, self.channels, self.w, self.h)).view(batch_size, feature_dim)
        return encoded_states


class KinematicsEncoder(nn.Module):
    def __init__(self, input_size):
        super(KinematicsEncoder, self).__init__()
        self.input_size = input_size

        # for CartPole
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, feature_dim),
        # ).to(device)

        # for Highway
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        ).to(device)

        # self.optimizer = optim.Adam(self.parameters(), lr=5e-4)  # Learning rate is set to 0.001
        # self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def encode(self, states):
        return self.forward(states)

    # def update_encoder(self):
    #     self.optimizer.zero_grad()
    #     # Compute the loss
    #     loss = self.loss_fn(output, target_tensor)
    #     # Backward pass: compute gradient of the loss with respect to model parameters
    #     loss.backward()
    #     # Update the parameters
    #     self.optimizer.step()
    #
    #     return loss.item()


class OccupancyGridEncoder(nn.Module):
    def __init__(self, input_size):
        super(OccupancyGridEncoder, self).__init__()
        self.input_size = input_size

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        ).to(device)

    def forward(self, x):
        return self.net(x)

    def encode(self, states):
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        return self.forward(states)


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)
