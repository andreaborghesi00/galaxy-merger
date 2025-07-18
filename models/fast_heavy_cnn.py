import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Dropout, ReLU, Sigmoid, GELU, Module, Sequential, Linear, Softmax, LeakyReLU
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary

class FastHeavyCNN(Module):
    def __init__(self):
        super(FastHeavyCNN, self).__init__()
        self.features = Sequential(
            self._make_layer(3, 32, 2),
            self._make_layer(32, 64, 2),
            self._make_layer(64, 128, 3),
            self._make_layer(128, 256, 3)
        )
        self.fc = Sequential(
            Linear(256*4*4, 1024),
            LeakyReLU(inplace=True),
            Dropout(0.25),
            Linear(1024, 512),
            LeakyReLU(inplace=True),
            Dropout(0.25),
            Linear(512, 1),
            Sigmoid()
        )

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding='same'))
            layers.append(BatchNorm2d(out_channels))
            layers.append(LeakyReLU(inplace=True))
        layers.append(MaxPool2d(2))
        return Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256*4*4)
        x = self.fc(x)
        return x
