import torch
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from time import time
from scipy import ndimage, misc

path = '/Users/romapatel/Documents/proto/models/cnn/'

class AlexNet(nn.Module):
    def __init__(self, image_size=64, num_classes=125):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        print x
        x = x.view(x.size(0), 256*7*7)
        x = self.classifier(x)
        return x


def run_alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model


if __name__ == '__main__':
    print 'Inside main!\n'
    run_alexnet()
