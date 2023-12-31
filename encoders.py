import torch
import torch.nn as nn
import torch.nn.functional as F
import constants
from enum import Enum

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")

kernel_size = 3
stride = 1
feature_size = 128


class ObservationType(Enum):
    Camera = 1
    Kinematics = 2


class CameraEncoder(nn.Module):
    def __init__(self, width, height, colors):
        super(CameraEncoder, self).__init__()
        self.conv1 = nn.Conv2d(colors, 16, kernel_size, stride).to(device)
        self.bn1 = nn.BatchNorm2d(16).to(device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride).to(device)
        self.bn2 = nn.BatchNorm2d(32).to(device)
        self.conv3 = nn.Conv2d(32, 32, kernel_size, stride).to(device)
        self.bn3 = nn.BatchNorm2d(32).to(device)
        self.w, self.h = width, height
        self.colors = colors

        def conv2d_size_out(size):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_w * conv_h * 32  # output size of the convolutional layers
        self.output = nn.Linear(linear_input_size, feature_size).to(device)
        self.norm = L2Norm()

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.norm(self.output(x.view(x.size(0), -1)))

    def encode(self, states):
        batch_size, k = states.size(0), states.size(1)
        encoded_states = self.forward(states.view(batch_size * k, self.colors, self.w, self.h)).view(batch_size, k, feature_size)
        return encoded_states


class KinematicsEncoder(nn.Module):
    def __init__(self, input_size):
        super(KinematicsEncoder, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.fc2 = nn.Linear(64, feature_size).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def encode(self, states):
        batch_size, k = states.size(0), states.size(1)
        states = states.view(batch_size * k, self.input_size)
        encoded_states = self.forward(states)
        encoded_states = encoded_states.view(batch_size, k, feature_size)
        return encoded_states


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)
