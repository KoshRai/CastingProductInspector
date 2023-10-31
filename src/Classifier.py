import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features = 1024),
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(out_features=256),
            nn.Sigmoid(),
            nn.Linear(in_features=256, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features = 128, out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=2),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)