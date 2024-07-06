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
    
def load_model(model_class, model_path, last=False, verbose=False):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_class (class): The class of the model to be loaded.
        model_path (str): The path to the model checkpoint file.
        last (bool, optional): Whether to load the last model state or the best model state. Defaults to False.
        verbose (bool, optional): Whether to print information about the loaded model. Defaults to False.

    Returns:
        model: The loaded model.
    """
    model = model_class()

    try:
        model_checkpoint = torch.load(model_path)
    except FileNotFoundError as e:
        print(f"Model checkpoint not found at {model_path}")
        raise e
    
    model.load_state_dict(model_checkpoint['model_state_dict' if last else 'best_model_state_dict'])
    if verbose:
        print(f"Loaded model: {model.__class__.__name__}")
        print(f"Performance history: {model_checkpoint['history']}")
    return model
    
if __name__ == "__main__":
    model_classes = [HeavyCNN, FastHeavyCNN, ResNet18, DeepMerge]
    for model_class in model_classes:
        model = model_class()
        print(f"Model: {model.__class__.__name__}")
        summary(model, (3, 75, 75))