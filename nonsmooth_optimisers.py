from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################### MODEL FREE #####################

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
        self.lstm = nn.LSTM(input_size=32*32*32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 128*128)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        return (2*torch.sigmoid(r_out2)-1).view(batch_size, 1, 128, 128)  # Modify this line to output a single channel


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
        self.lstm = nn.LSTM(input_size=32*32*32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 128*128+1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        x = r_out2[:, 0:128*128]
        x = (2*torch.sigmoid(x)-1).view(batch_size, 1, 128, 128)
        tau = torch.nn.Softplus()(r_out2[:, 128*128])
        return x, tau



class CNN_LSTM_Full(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Full, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=32*32*32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 128*128)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        return torch.sigmoid(r_out2).view(batch_size, 1, 128, 128)  # Modify this line to output a single channel



#################### MODEL BASED #####################




class TAU_NET(nn.Module):
    def __init__(self):
        super(TAU_NET, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=32*32*32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        x = self.fc(r_out[:, -1, :])
        tau = torch.nn.Softplus()(x)
        return tau

## PGD just uses a CNN_LSTM




