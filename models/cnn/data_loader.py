import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
#from skimage import color
from scipy import ndimage, misc
import os
import numpy as np
import codecs
from PIL import Image

path = '/Users/romapatel/Documents/proto/'
path = '/nlp/data/romap/proto/'
model_path = path + 'models/cnn/'

class SketchData(Dataset):
    categories = [name for name in os.listdir(path + 'data/sketchy/categories/') if '.DS' not in name]
    class_to_idx = {_class: i for i, _class in enumerate(categories)}
    
    train_dir = path + 'data/sketchy/train/'
    test_dir = path + 'data/sketchy/test/'
    print class_to_idx

    def __init__(self, root, train=True, transform=None, target_transform=None, img_type='sketch'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.categories = [name for name in os.listdir(path + 'data/sketchy/categories/') if '.DS' not in name]
        self.categories = self.categories



        self.img_type= img_type
        self.training_file = 'data/sketchy/processed/training.pt'
        self.test_file = 'data/sketchy/processed/test.pt'

        
        if self.train:
            self.train_data, self.train_labels = torch.load(os.path.join(self.root, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(self.root, self.test_file))
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):

        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))


    def __getitem__(self, idx):
        # flipped
        if self.train:
            target, img = self.train_data[idx], self.train_labels[idx]
        else:
            target, img = self.test_data[idx], self.test_labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def get_data(self, split):
        images, labels = [], []
        class_to_idx = {_class: i for i, _class in enumerate(self.categories)}
        for category in self.categories:
            if split is True:
                f = open(path + 'data/sketchy/categories/' + category + '/' + self.img_type + '/train.txt', 'r')
            else:
                f = open(path + 'data/sketchy/categories/' + category + '/' + self.img_type + '/test.txt', 'r')
            lines = f.readlines()[1:]
            
            for line in lines:
                #img = ndimage.imread(path + 'data/sketchy/figs/' + line.strip())
                img = Image.open(path + 'data/sketchy/figs/' + line.strip())
                #img = img.resize((64, 64))

                #img = color.rgb2gray(img)
                labels.append(class_to_idx[category])
                #parsed = np.frombuffer(img, dtype=np.float)
                images.extend(np.array(img).astype(np.uint8))
        # returns labels, image tensors
        return torch.from_numpy(np.array(labels)).view(len(labels)).long(), torch.from_numpy(np.array(images)).view(len(labels), 3, 256, 256)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

# if data doesn't exist already, create torch dataset files
def create_dataset():
    s = SketchData(path, True, None, None, 'sketch')
    train_labels, train_images = s.get_data(True)
    test_labels, test_images = s.get_data(False)

    train_set = (train_labels, train_images)
    test_set = (test_labels, test_images)


    with open(path + 'data/sketchy/processed/training.pt', 'wb') as f:
        torch.save(train_set, f)
    with open(path + 'data/sketchy/processed/test.pt', 'wb') as f:
        torch.save(test_set, f)
    
if __name__ == '__main__':
    print 'Inside main!\n'

    create_dataset()

    
