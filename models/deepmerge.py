import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Dropout, ReLU, Sigmoid, GELU, Module, Sequential, Linear, Softmax, LeakyReLU
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary

class DeepMerge(Module):
    def __init__(self):
        super(DeepMerge, self).__init__()
        self.conv1 = Sequential(
            Conv2d(3, 8, kernel_size=5, padding=2),
            ReLU(),
            BatchNorm2d(8),
            MaxPool2d(2),
            Dropout(0.5)
        )  # output size 37x37

        self.conv2 = Sequential(
            Conv2d(8, 16, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm2d(16),
            MaxPool2d(2),
            Dropout(0.5)
        )  # output size 18x18

        self.conv3 = Sequential(
            Conv2d(16, 32, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(2),
            Dropout(0.5)
        )  # output size 9x9

        self.fc1 = Sequential(
            Linear(32*9*9, 64),
            Softmax(dim=1),
            Linear(64, 32),
            Softmax(dim=1),
            Linear(32, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x
