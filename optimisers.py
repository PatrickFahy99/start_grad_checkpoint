
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import torch
import random
from torch.fft import fftn, ifftn
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class TauFuncNet(nn.Module):
    def __init__(self):
        super(TauFuncNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((29, 29))  # Ensure output is always 29x29
        self.fc1 = None  # Will be defined later based on input size
        self.fc2 = nn.Linear(120 + 2, 84)  # +2 for the two additional float inputs
        self.fc3 = nn.Linear(84, 1)
        self.float1_layer = nn.Linear(1, 1)  # Layers to process the float inputs
        self.float2_layer = nn.Linear(1, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dynamically define fc1 based on input size if not defined
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1) + 2, 120 + 2).to(x.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.nn.Softplus()(self.fc3(x))
        return x


class TauFuncNetL(nn.Module):
    def __init__(self):
        super(TauFuncNetL, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((29, 29))  # Ensure output is always 29x29
        self.fc1 = None  # Will be defined later based on input size
        self.fc2 = nn.Linear(120 + 2, 84)  # +2 for the two additional float inputs
        self.fc3 = nn.Linear(84, 1)
        self.float1_layer = nn.Linear(1, 1)  # Layers to process the float inputs
        self.float2_layer = nn.Linear(1, 1)

    def forward(self, x, float1, float2, L):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dynamically define fc1 based on input size if not defined
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1) + 2, 120 + 2).to(x.device)

        # Process float inputs
        float1 = F.relu(self.float1_layer(float1.float().unsqueeze(-1)))
        float2 = F.relu(self.float2_layer(float2.float().unsqueeze(-1)))

        x = torch.cat((x, float1.unsqueeze(0), float2.unsqueeze(0)), dim=-1)  # Concatenate on the last dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (2/L) * torch.sigmoid(self.fc3(x))
        return x


class TauFunc10Net(nn.Module):
    def __init__(self):
        super(TauFunc10Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((29, 29))  # Ensure output is always 29x29
        self.fc1 = None  # Will be defined later based on input size
        self.fc2 = nn.Linear(120 + 2, 84)  # +2 for the two additional float inputs
        self.fc3 = nn.Linear(84, 1)
        self.float1_layer = nn.Linear(1, 1)  # Layers to process the float inputs
        self.float2_layer = nn.Linear(1, 1)

    def forward(self, x, float1, float2):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dynamically define fc1 based on input size if not defined
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1) + 2, 120 + 2).to(x.device)

        # Process float inputs
        float1 = F.relu(self.float1_layer(float1.float().unsqueeze(-1)))
        float2 = F.relu(self.float2_layer(float2.float().unsqueeze(-1)))

        x = torch.cat((x, float1.unsqueeze(0), float2.unsqueeze(0)), dim=-1)  # Concatenate on the last dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = 10 * torch.sigmoid(self.fc3(x))
        return x


#
# class TauFuncUnboundedAboveNet(nn.Module):
#     def __init__(self):
#         super(TauFuncUnboundedAboveNet, self).__init__()
#
#         # Define the CNN layers
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 6, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(6, 16, 5),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((29, 29))
#         )
#
#         # Define the LSTM layer
#         self.lstm = nn.LSTM(input_size=16 * 29 * 29, hidden_size=64, num_layers=1, batch_first=True)
#
#         # Define the fully connected layers
#         self.fc1 = nn.Linear(64, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 1)  # Output with linear activation
#
#     def forward(self, x):
#         # Pass input through CNN layers
#         x = self.cnn(x)
#
#         # Reshape the output for LSTM input
#         x = x.view(x.size(0), 1, -1)  # Adding a time dimension
#
#         # Pass through LSTM
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]  # Take the output from the last time step
#
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = (50/27)*F.sigmoid(self.fc3(x))  # Linear activation for an unbounded above output
#
#         return x


class TauFuncUnboundedAboveNet(nn.Module):
    def __init__(self):
        super(TauFuncUnboundedAboveNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((29, 29))
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # Output with linear activation

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation for an unbounded above output

        return x

#
# class TauBetaFunc(nn.Module):
#     def __init__(self):
#         super(TauBetaFunc, self).__init__()
#
#         # Define the CNN layers
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 6, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(6, 16, 5),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((29, 29))
#         )
#
#         # Define the LSTM layer
#         self.lstm = nn.LSTM(input_size=16 * 29 * 29, hidden_size=64, num_layers=1, batch_first=True)
#
#         # Define the fully connected layers for tau
#         self.fc1_tau = nn.Linear(64, 120)
#         self.fc2_tau = nn.Linear(120, 84)
#         self.fc3_tau = nn.Linear(84, 1)  # Output for tau
#
#         # Define the fully connected layers for beta
#         self.fc1_beta = nn.Linear(64, 120)
#         self.fc2_beta = nn.Linear(120, 84)
#         self.fc3_beta = nn.Linear(84, 1)  # Output for beta
#
#     def forward(self, x):
#         # Pass input through CNN layers
#         x = self.cnn(x)
#
#         # Reshape the output for LSTM input
#         x = x.view(x.size(0), 1, -1)  # Adding a time dimension
#
#         # Pass through LSTM
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]  # Take the output from the last time step
#
#         # Fully connected layers for tau
#         x_tau = F.relu(self.fc1_tau(x))
#         x_tau = F.relu(self.fc2_tau(x_tau))
#         tau = (50/27)*F.sigmoid(self.fc3_tau(x_tau))
#
#         # Fully connected layers for beta
#         x_beta = F.relu(self.fc1_beta(x))
#         x_beta = F.relu(self.fc2_beta(x_beta))
#         beta = torch.nn.Sigmoid()(self.fc3_beta(x_beta))
#
#         return tau, beta



class TauBetaFunc(nn.Module):
    def __init__(self):
        super(TauBetaFunc, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((29, 29))
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3_tau = nn.Linear(84, 1)  # Output for tau
        self.fc3_beta = nn.Linear(84, 1)  # Output for beta

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        tau = torch.nn.Softplus()(self.fc3_tau(x))
        beta = torch.nn.Sigmoid()(self.fc3_beta(x))

        return tau, beta


class TauBetaFunc(nn.Module):
    def __init__(self, input_channels):
        super(TauBetaFunc, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((29, 29))
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3_tau = nn.Linear(84, 1)  # Output for tau
        self.fc3_beta = nn.Linear(84, 1)  # Output for beta

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        tau = torch.nn.Softplus()(self.fc3_tau(x))
        beta = torch.nn.Sigmoid()(self.fc3_beta(x))

        return tau, beta





class TauFuncUnboundedNet(nn.Module):
    def __init__(self):
        super(TauFuncUnboundedNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((29, 29))  # Ensure output is always 29x29
        self.fc1 = None  # Will be defined later based on input size
        self.fc2 = nn.Linear(120 + 2, 84)  # +2 for the two additional float inputs
        self.fc3 = nn.Linear(84, 1)
        self.float1_layer = nn.Linear(1, 1)  # Layers to process the float inputs
        self.float2_layer = nn.Linear(1, 1)

    def forward(self, x, float1, float2):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dynamically define fc1 based on input size if not defined
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1) + 2, 120 + 2).to(x.device)

        # Process float inputs
        float1 = F.relu(self.float1_layer(float1.float().unsqueeze(-1)))
        float2 = F.relu(self.float2_layer(float2.float().unsqueeze(-1)))

        x = torch.cat((x, float1.unsqueeze(0), float2.unsqueeze(0)), dim=-1)  # Concatenate on the last dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class UnrollingFunc(nn.Module):
    def __init__(self, iters=10, hidden_size=100):
        super(UnrollingFunc, self).__init__()

        # MLP to learn the taus as a function of sig and std
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),  # Input size of 2 for sig and std
            nn.ReLU(),
            nn.Linear(hidden_size, iters)  # Output size of 'iters' for the taus
        )

    def forward(self, sig, std):
        # Combine sig and std into a single tensor of size [batch_size, 2]
        input_tensor = torch.stack([sig, std], dim=-1)

        # Pass through MLP
        #self.taus = 2*torch.sigmoid(self.mlp(input_tensor))
        self.taus = self.mlp(input_tensor)

        return self.taus




class Unrolling(nn.Module):
    def __init__(self, iters=10):
        super(Unrolling, self).__init__()
        self.taus = nn.Parameter(torch.rand(iters))  # Learnable tau values

    def forward(self, x):
        return self.tausF

class FixedModel(nn.Module):
    def __init__(self):
        super(FixedModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 1)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.nn.Softplus()(self.fc4(x.float()))  # Output layer (no activation function as we're doing regression)
        return x


class FixedModelSoftplus(nn.Module):
    def __init__(self):
        super(FixedModelSoftplus, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 1)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.nn.Softplus()(self.fc4(x.float()))  # Output layer (no activation function as we're doing regression)
        return x


class FixedModelZeroOne(nn.Module):
    def __init__(self):
        super(FixedModelZeroOne, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 1)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.nn.sigmoid()(self.fc4(x.float()))  # Output layer (no activation function as we're doing regression)
        return x


class FixedModelL(nn.Module):
    def __init__(self):
        super(FixedModelL, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 1)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x, L):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = (2/L)*torch.nn.sigmoid()(self.fc4(x.float()))  # Output layer (no activation function as we're doing regression)
        return x



class FixedMomentumModel(nn.Module):
    def __init__(self):
        super(FixedMomentumModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 2)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = self.fc4(x.float())  # Output layer (no activation function as we're doing regression)
        x0 = torch.nn.Softplus()(x[0])
        x1 = torch.nn.Sigmoid()(x[1])

        # Create a new tensor to store the results
        x = torch.stack([x0, x1], dim=0)
        return x


class AdagradModel(nn.Module):
    def __init__(self):
        super(AdagradModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 2)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = self.fc4(x.float())  # Output layer (no activation function as we're doing regression)
        x0 = torch.nn.Sigmoid()(x[0])#torch.nn.Softplus()(x[0])
        x1 = torch.nn.Sigmoid()(x[1])
        # Create a new tensor to store the results
        x = torch.stack([x0, x1], dim=0)
        return x
class RMSPropModel(nn.Module):
    def __init__(self):
        super(RMSPropModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 3)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = self.fc4(x.float())  # Output layer (no activation function as we're doing regression)
        x0 = torch.nn.Sigmoid()(x[0])#torch.nn.Softplus()(x[0])
        x1 = torch.nn.Sigmoid()(x[1])
        #x2 = torch.tensor(1e-08).to(device)#torch.nn.Sigmoid()(x[2])
        x2 = torch.nn.Sigmoid()(x[2])
        # Create a new tensor to store the results
        x = torch.stack([x0, x1, x2], dim=0)
        return x



class AdamModel(nn.Module):
    def __init__(self):
        super(AdamModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input nodes (for sigma and delta), 100 nodes in hidden layer
        self.fc2 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc3 = nn.Linear(10, 10)  # 100 nodes in hidden layer, 1 output node (for tau)
        self.fc4 = nn.Linear(10, 4)  # 100 nodes in hidden layer, 1 output node (for tau)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Apply ReLU activation function after first layer
        x = torch.relu(self.fc2(x.float()))  # Output layer (no activation function as we're doing regression)
        x = torch.relu(self.fc3(x.float()))  # Output layer (no activation function as we're doing regression)
        x = self.fc4(x.float())  # Output layer (no activation function as we're doing regression)
        x0 = torch.nn.Sigmoid()(x[0])#torch.nn.Softplus()(x[0])
        x1 = torch.nn.Sigmoid()(x[1])
        x2 = torch.nn.Sigmoid()(x[2])
        #x3 = torch.tensor(1e-08).to(device)#torch.nn.Sigmoid()(x[3])
        x3 = torch.nn.Softplus()(x[3])
        # Create a new tensor to store the results
        x = torch.stack([x0, x1, x2, x3], dim=0)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn

class UpdateModel(nn.Module):
    def __init__(self):
        super(UpdateModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Sigmoid(),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x, sig, std):
        float1 = torch.full((128, 128), sig).unsqueeze(0).unsqueeze(0).to(device)
        float2 = torch.full((128, 128), std).unsqueeze(0).unsqueeze(0).to(device)
        x = torch.cat((x, float1, float2), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=32*64*64, hidden_size=128, num_layers=1, batch_first=True)  # Adjust input_size
        self.fc = nn.Linear(128, 256*256)  # Adjust output size

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        return (2*torch.sigmoid(r_out2)-1).view(batch_size, 1, 256, 256)  # Adjust output shape


import torch.nn as nn
import torch


class CNN_LSTM_CORRECTION(nn.Module):
    def __init__(self):
        super(CNN_LSTM_CORRECTION, self).__init__()

        # Feature extraction for each input image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjust input channels (3 for RGB)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        # Combining features
        self.fc_combine = nn.Linear(128 * 64 * 64 * 4, 256)  # Adjust input size

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)

        # Further processing
        self.fc_process = nn.Linear(128, 128)

        # Image generation
        self.cnn_generate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output 3 channels for RGB
            nn.Tanh(),  # Adjust activation function based on your needs
        )

        # Scalar prediction
        self.fc_scalar = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, num_timesteps, C, H, W = x.size()

        # Extract features from each input image for each timestep
        features = []
        for t in range(num_timesteps):
            img_t = x[:, t, :, :, :]  # Select the input at timestep t
            features_t = self.cnn(img_t)  # Extract features
            features.append(features_t)

        # Combine features across timesteps (e.g., average pooling)
        combined_features = torch.mean(torch.stack(features), dim=0)

        # Combine features across channels
        combined_features = combined_features.view(batch_size, -1)

        # Further process the combined features
        processed_features = self.fc_process(
            self.lstm(combined_features.view(batch_size, self.num_timesteps, -1))[0][:, -1, :])

        # Generate the image
        generated_image = self.cnn_generate(processed_features.view(-1, 128, 1, 1))

        # Predict the scalar value
        scalar_prediction = self.fc_scalar(processed_features)

        return generated_image, scalar_prediction


class CNN_LSTM_CORRECTION(nn.Module):
    def __init__(self):
        super(CNN_LSTM_CORRECTION, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=32*64*64, hidden_size=128, num_layers=1, batch_first=True)  # Adjust input_size
        self.fc = nn.Linear(128, 256*256+1)  # Adjust output size

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        x = r_out2[:, 0:256*256]
        x = (2*torch.sigmoid(x)-1).view(batch_size, 1, 256, 256)
        tau = torch.nn.Softplus()(r_out2[:, 256*256])
        return x, tau



class CNN_LSTM_Full(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Full, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=32*64*64, hidden_size=128, num_layers=1, batch_first=True)  # Adjust input_size
        self.fc = nn.Linear(128, 256*256)  # Adjust output size

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        return torch.sigmoid(r_out2.view(batch_size, 1, 256, 256))  # Adjust output shape


class GeneralUpdateModel(nn.Module):
    def __init__(self):
        super(GeneralUpdateModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Sigmoid(),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x, sig, std):
        float1 = torch.full((128, 128), sig).unsqueeze(0).unsqueeze(0).to(device)
        float2 = torch.full((128, 128), std).unsqueeze(0).unsqueeze(0).to(device)
        x = torch.cat((x, float1, float2), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


