import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Dropout, ReLU, Sigmoid, GELU, Module, Sequential, Linear, Softmax, LeakyReLU
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary

class HeavyCNN(Module):
    def __init__(self):
        super(HeavyCNN, self).__init__()
        self.conv1 = Sequential(
            Conv2d(3, 32, kernel_size=6, padding='same'),
            BatchNorm2d(32),
            GELU(),
            Conv2d(32, 32, kernel_size=6, padding='same'),
            BatchNorm2d(32),
            GELU(),
            MaxPool2d(2)
        )

        self.conv2 = Sequential(
            Conv2d(32, 64, kernel_size=5, padding='same'),
            BatchNorm2d(64),
            GELU(),
            Conv2d(64, 64, kernel_size=5, padding='same'),
            BatchNorm2d(64),
            GELU(),
            MaxPool2d(2)
        )
        
        self.conv3 = Sequential(
            Conv2d(64, 128, kernel_size=3, padding='same'),
            BatchNorm2d(128),
            GELU(),
            Conv2d(128, 128, kernel_size=3, padding='same'),
            BatchNorm2d(128),
            GELU(),
            Conv2d(128, 128, kernel_size=3, padding='same'),
            BatchNorm2d(128),
            GELU(),
            MaxPool2d(2)
        )

        self.conv4 = Sequential(
            Conv2d(128, 256, kernel_size=3, padding='same'),
            BatchNorm2d(256),
            GELU(),
            Conv2d(256, 256, kernel_size=3, padding='same'),
            BatchNorm2d(256),
            GELU(),
            Conv2d(256, 256, kernel_size=3, padding='same'),
            BatchNorm2d(256),
            GELU(),
            MaxPool2d(2)
        )

        self.fc1 = Sequential(
            Linear(256*4*4, 1024),
            GELU(),
            Dropout(0.25),
            Linear(1024, 512),
            GELU(),
            Dropout(0.25),
            Linear(512, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 256*4*4)
        x = self.fc1(x)
        return x
    