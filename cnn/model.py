from typing import OrderedDict
import torch
from torch.nn import AdaptiveAvgPool2d, AvgPool2d, Conv2d, Flatten, Linear, Sequential
from torch.nn.modules import MaxPool2d, ReLU

class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()

        layer1 = Sequential(
            Conv2d(3, 64, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(64, 64, (3, 3), stride=1, padding="same"),
            ReLU(),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )

        layer2 = Sequential(
            Conv2d(64, 128, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(128, 128, (3, 3), stride=1, padding="same"),
            ReLU(),
            MaxPool2d((2, 2), stride=2)
        )

        layer3 = Sequential(
            Conv2d(128, 256, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(256, 256, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(256, 256, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(256, 256, (3, 3), stride=1, padding="same"),
            ReLU(),
            MaxPool2d((2, 2), stride=2)
        )

        layer4 = Sequential(
            Conv2d(256, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            MaxPool2d((2, 2), stride=2)
        )

        layer5 = Sequential(
            Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            ReLU(),
            MaxPool2d((2, 2), stride=2),
            AdaptiveAvgPool2d((7, 7)),
            Flatten()
        )

        classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, 10)
        )

        self.features = Sequential(
            layer1, layer2, layer3, layer4, layer5, classifier
        )

    def forward(self, x):
        return self.features.forward(x)
