from __future__ import print_function
import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse
import numpy as np

X_dim, N, z_dim = 300, 200, 100
# encoder
class Q_net(nn.Module):
    
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
        
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        
        return xgauss

# decoder
class P_net(nn.Module):
    
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
        
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)

        return F.sigmoid(x)

# discriminator
class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
        
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        
        return F.sigmoid(self.lin3(x))

if __name__ == '__main__':
    torch.manual_seed(10)
    Q = Q_net()
    P = P_net()
    D_gauss = D_net_gauss()                # Discriminator adversarial

    '''
    if torch.cuda.is_available():
        Q = Q.cuda()
        P = P.cuda()
        D_cat = D_gauss.cuda()
        D_gauss = D_net_gauss().cuda()
    '''
    # Set learning rates
    gen_lr, reg_lr = 0.0006, 0.0008
    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)
    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

    X = torch.randn(N, 300)
    print(X)
    print(Variable(X))
    
    z_sample = Q(X)
    X_sample = P(z_sample)
    recon_loss = F.binary_cross_entropy(X_sample + TINY, 
                                        X.resize(train_batch_size, X_dim) + TINY)
    recon_loss.backward()
    P_decoder.step()
    Q_encoder.step()

    Q.eval()    
    z_real_gauss = Variable(torch.randn(train_batch_size, z_dim) * 5)   # Sample from N(0,5)
    if torch.cuda.is_available():
        z_real_gauss = z_real_gauss.cuda()
    z_fake_gauss = Q(X)

    # Compute discriminator outputs and loss
    D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)
    D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
    D_loss.backward()       # Backpropagate loss
    D_gauss_solver.step()   # Apply optimization step

    # Generator
    Q.train()   # Back to use dropout
    z_fake_gauss = Q(X)
    D_fake_gauss = D_gauss(z_fake_gauss)

    G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
    G_loss.backward()
    Q_generator.step()
