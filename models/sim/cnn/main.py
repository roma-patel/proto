import torch
import argparse
import time
from time import time
from scipy import ndimage, misc
from alexnet import AlexNet
from data_loader import SketchData
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable

path = '/Users/romapatel/Documents/proto/'
device = 'cpu'; model_names = ['alexnet']

parser = argparse.ArgumentParser(description='SketchCNN')

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
    print 'Inside run!'
    global args
    args = parser.parse_args()
    print args
    # add checkpoint resume option
    # load datasets
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
    model = AlexNet()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best_prec = 0
    for epoch in range(args.epochs):
        print 'Epoch: ' + str(epoch)
        adjust_learning_rate(optimizer, epoch)
        print 'Adjusted learning rate'
        train(train_loader, model, criterion, optimizer, epoch)
        print 'Trained!'
        precision = validate(val_loader, model, criterion)
        print 'Got precision!'
        best_prec = max(precision.data[0], best_prec)
        print 'Updated best precision!'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_prec1': best_prec,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, (precision.data[0] > best_prec))

def train(train_loader, model, criterion, optimizer, epoch):

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
    run()
