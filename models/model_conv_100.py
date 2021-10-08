import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ConvNet(nn.Module):
    def __init__(self):
      super(ConvNet, self).__init__()
      self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64,
                             kernel_size = 5)

      self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64,
                             kernel_size = 5)

      self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

      self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64,
                             kernel_size = 3)

      self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64,
                             kernel_size = 3)

      self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

      self.fc1 = nn.Linear(1024, 128, bias = True)
      self.fc2 = nn.Linear(128, 100, bias = True)

      self.dropout25 = nn.Dropout(0.25)
      self.dropout50 = nn.Dropout(0.5)

    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.pool1(x)

      x = self.conv3(x)
      x = F.relu(x)
      x = self.conv4(x)
      x = F.relu(x)
      x = self.pool2(x)

      x = x.view(x.shape[0], -1)

      x = self.fc1(x)
      self.h = x.clone()
      x = self.dropout50(x)
      x = F.relu(x)

      x = self.fc2(x)

      #output = F.log_softmax(x, dim=1)
      return x
