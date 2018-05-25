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


path = '/Users/romapatel/Documents/proto/models/ae/'

parser = argparse.ArgumentParser(description="sketch image params")
parser.add_argument('num_epochs', metavar='N', type=int, help="Number of epochs")
parser.add_argument('batch_size', metavar='N', type=int, help="Batch size")
parser.add_argument('hidden_size', metavar='N', type=int, help="Hidden dimension")
parser.add_argument('image_size', metavar='N', type=int, help="Image height (hxh)")

args = parser.parse_args()
batch_size, hidden_size, num_epochs = args.batch_size, args.hidden_size, args.num_epochs
img_size = args.image_size

img = ndimage.imread('/Users/romapatel/Desktop/sketches_png/airplane/1.png')
print img


# MNIST dataset
'''
dataset = dsets.MNIST(root=path+'convert_MNIST/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
'''
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
dataset = dsets.ImageFolder(
        path+'data/sketches/',
        transforms.Compose([
            transforms.Resize(1111),

            transforms.ToTensor(),
        ]))

print dataset



# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Autoencoder(nn.Module):
    def __init__(self, in_dim=None, h_dim=None):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU()
            )

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, in_dim),
            nn.Sigmoid()
            )


    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        print 'insider forward'
        print 'x: '
        print x
        out = self.encoder(x)
        out = self.decoder(out)
        return out


ae = Autoencoder(in_dim=img_size*img_size, h_dim=hidden_size)

if torch.cuda.is_available():
    ae.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

# save fixed inputs for debugging
fixed_x, _ = next(data_iter)

torchvision.utils.save_image(Variable(fixed_x).data.cpu(), path + '/data/real_images.png')
fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1))

    
for epoch in range(num_epochs):
    print 'epoch: ' + str(epoch)
    t0 = time()
    for i, (images, _) in enumerate(data_loader):
        print 'i: '; print i
        print 'images: '; print images
        # flatten the image
        images = to_var(images.view(-1, img_size*img_size))
        print 'size'
        print images.size(0)

        print images.size(1)
        out = ae(images)
        loss = criterion(out, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs' 
                %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.data[0], time()-t0))

    # save the reconstructed images
    reconst_images = ae(fixed_x)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, img_size, img_size)
    torchvision.utils.save_image(reconst_images.data.cpu(), path + '/data/reconst_images_%d.png' % (epoch+1))
