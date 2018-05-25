import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from time import time

path = '/Users/romapatel/Documents/proto/models/ae/'
num_epochs, batch_size, hidden_size = 50, 100, 100


# MNIST dataset
'''
dataset = dsets.MNIST(root=path+'convert_MNIST/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
'''
#dataset = dsets.DatasetFolder(root=path+'data/sketches/', loader=loader, extensions=['png', 'jpg'], transform=transforms.ToTensor())
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
dataset = dsets.ImageFolder(
        path+'data/sketches/',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

'''
#Sketch dataset
class SketchDataset(Dataset):

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
'''
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

