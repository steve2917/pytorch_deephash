import os
import torch
import torch.nn as nn
from torchvision import models
import math
import torch.utils.model_zoo as model_zoo

os.environ['TORCH_HOME'] = 'models'
alexnet_model = models.alexnet(pretrained=True)
resnet_model = models.resnet50(pretrained=True)

class AlexNetPlusLatent(nn.Module):
    def __init__(self, bits):
        super(AlexNetPlusLatent, self).__init__()
        self.bits = bits
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.bits, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        return features, result

class ResNetPlusLatent(nn.Module):
    def __init__(self, bits):
        super(ResNetPlusLatent, self).__init__()
        self.bits = bits
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool

        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.bn1.parameters():
            param.requires_grad = False

        for param in self.layer1.parameters():
            param.requires_grad = False

        for param in self.layer2.parameters():
            param.requires_grad = False

        for param in self.layer3.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool
        )

        self.Linear1 = nn.Linear(2048, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.bits, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2048)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        return features, result