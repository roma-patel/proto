# -*- coding: utf-8 -*-
import torch
import argparse
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from time import time
from scipy import ndimage, misc
from alexnet import AlexNet
from data_loader import SketchData

path = '/Users/romapatel/Documents/proto/'
path = '/nlp/data/romap/proto/'

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model_names = ['alexnet']
parser = argparse.ArgumentParser(description='CNN for Sketches')


# -epochs 90 -b 256 -lr 0.1 -momentum 0.9 -p 10 
parser.add_argument('-epochs', default=90, type=int, metavar='N',
                    help='Number of epochs to run')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-b', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '-evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def run():
    global args; args = parser.parse_args()

    if args.resume:
        print 'Resuming from checkpoint!'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # load datasets, split train and test
    '''
    train_dataset = SketchData(root=path,
                            train=True, 
                            transform=transforms.ToTensor(),
                            target_transform=transforms.ToTensor(),
                            )
    test_dataset = SketchData(root=path,
                            train=False, 
                            transform=transforms.ToTensor(),
                            target_transform=transforms.ToTensor(),
                            )
    '''
    train_dataset = SketchData(root=path,
                            train=True, 
                            transform=None,
                            target_transform=None,
                            )
    val_dataset = SketchData(root=path,
                            train=False, 
                            transform=None,
                            target_transform=None,
                            )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.b, 
                                           shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=args.b, 
                                          shuffle=False)
    
    # create model, set parameters, optimiser, loss
    model = AlexNet()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best_prec = 0;
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        precision = validate(val_loader, model, criterion)
        best_prec = max(precision.data[0], best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_prec1': best_prec,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, (precision.data[0] > best_prec))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model = model.double()
    model.train()

    #end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end)

        #target = target.cuda(non_blocking=True)

        # compute output
        input = Variable(input)
        target = Variable(target)
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        #losses.update(loss.item(), input.size(0))

        losses.update(loss, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        #batch_time.update(time.time() - end)
        #end = time.time()

        '''
        if i % args.p == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
        '''
        print 'Epoch ' + str(epoch)
        print 'Loss ' + str(losses.val) + ', ' + str(losses.avg)
        print 'Prec@1 ' + str(top1.val)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval(); no_grad = True

    if no_grad is True:
        #end = time.time()
        for i, (input, target) in enumerate(val_loader):
            #target = target.cuda(non_blocking=True)

            # compute output
            input = Variable(input)
            target = Variable(target)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            #losses.update(loss.item(), input.size(0))
            losses.update(loss, input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()

            '''
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            '''
            
        #print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
#              .format(top1=top1, top5=top5))
        print 'VAL SET PREC: ' + str(top1.avg)
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    no_grad = True
    """Computes the precision@k for the specified values of k"""
    if no_grad is True:
        maxk = max(topk)
        batch_size = target.size(0)
        print batch_size

        print maxk;maxk = 3
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
if __name__ == '__main__':
    print 'Inside main!\n'
    run()
    print 'Finished run!\n'
    
