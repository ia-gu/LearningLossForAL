'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random
import csv
import matplotlib.pyplot as plt

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10

# Utils
# import visdom
from tqdm import tqdm

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
from test import test
from config import *
from trainer import Trainer
from data.sampler import SubsetSequentialSampler
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=9999)
args = parser.parse_args()
SEED = args.seed
# Seed
random_seed = SEED
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


# Logs
log_path = 'logs/lr_' + str(LR) + 'momentum_' + str(MOMENTUM) + 'batch_' + str(BATCH) + '/' + str(NAME) + str(SEED) + '/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(log_path+'hist/', exist_ok=True)
logging.basicConfig(level=logging.ERROR, filename=log_path+'result.txt', format='%(message)s')

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar10_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
cifar10_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
cifar10_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)

def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


##
# Main
if __name__ == '__main__':

    for trial in range(TRIALS):
        with open(log_path+'/loss_prediction.csv', mode='a') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial'+str(trial+1)])
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]
        unlabeled_set = indices[ADDENDUM:]
        
        train_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(cifar10_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        resnet18    = resnet.ResNet18(num_classes=10).cuda()
        loss_module = lossnet.LossNet().cuda()
        models      = {'backbone': resnet18, 'module': loss_module}
        ACC = []
        hist = [0]*10
        models['backbone'].cuda()

        trainer = Trainer(log_path)

        # Active learning cycles
        for cycle in range(CYCLES):
            print(f'reset weight')
            models['backbone'] = resnet.ResNet18(num_classes=10).cuda()

            for _, labels in train_loader:
                for i in labels:
                    hist[i] += 1
            fig = plt.figure()
            plt.bar(classes, hist, width=0.9)
            plt.xlabel('classes')
            plt.ylabel('number of queried data')
            fig.savefig(f'{log_path}hist/trial_{str(trial)}cycle_{str(cycle)}.png')

            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.CosineAnnealingLR(optim_backbone, EPOCH)
            sched_module   = lr_scheduler.CosineAnnealingLR(optim_module, EPOCH)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            trainer.train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL,)
            acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            logging.error('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            ACC.append(acc)

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)
            
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH, 
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
        
        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))
        fig = plt.figure()
        plt.plot(ACC)
        plt.title('accuracy')
        plt.ylabel('test Accuracy')
        plt.xlabel('Iteration')
        plt.grid(True)
        fig.savefig(log_path+'acc'+str(trial+1)+'.png')
        with open(log_path+'test'+str(trial+1)+'.csv', mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(ACC)