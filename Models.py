from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Dropout, ReLU, Sigmoid, GELU, Module, Sequential, Linear, Softmax
from torchvision.models import resnet18, ResNet18_Weights

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