import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Dropout, ReLU, Sigmoid, GELU, Module, Sequential, Linear, Softmax, LeakyReLU
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary

class ClassifierHead(Module):
    def __init__(self, in_features):
        super(ClassifierHead, self).__init__()
        self.fc1 = Sequential(
            Linear(in_features, 1024),
            GELU(),
            Dropout(0.5),
            Linear(1024, 512),
            GELU(),
            Dropout(0.5),
            Linear(512, 1),
            Sigmoid()
        )

    def forward(self, x):
        return self.fc1(x)


class ResNet18(Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = ClassifierHead(in_features)

        unfreeze_layers = ['layer3','layer4', 'fc']
        for name, param in self.resnet.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)
    
    
