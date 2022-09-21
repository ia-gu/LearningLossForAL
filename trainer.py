from tqdm import tqdm
import torch
from config import *
import csv
import logging
import os
from test import test
class Trainer:
    def __init__(self, log_path):
        self.log_path = log_path
        pass

    def LossPredLoss(self, input, target, margin=1.0, reduction='mean'):
        # assertion: confirm the conditions
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape
        
        input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)//2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
        
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0) # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()
        
        return loss

    def train_epoch(self, models, criterion, optimizers, dataloaders, epoch, epoch_loss,):
        models['backbone'].train()
        models['module'].train()
        total = 0
        correct = 0

        for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
            inputs = data[0].cuda()
            labels = data[1].cuda()

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            # ResNet18は出力scoresと中間層の特徴量リストfeaturesを返す
            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if epoch > epoch_loss:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            # Loss prediction
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss   = self.LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss            = m_backbone_loss + WEIGHT * m_module_loss
            if(epoch==epoch_loss):
                Loss = []
                for i in target_loss:
                    Loss.append(i.item())
                Pred_Loss = []
                for i in pred_loss:
                    Pred_Loss.append(i.item())
                with open(self.log_path+'loss_prediction.csv', mode='a') as f:
                    writer = csv.writer(f)
                    writer.writerow(Loss)
                    writer.writerow(Pred_Loss)
                    writer.writerow([])

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()

        return correct/total

    def train(self, models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss,):
        print('>> Train a Model.')
        best_acc = 0.
        checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        for epoch in range(num_epochs):
            schedulers['backbone'].step()
            schedulers['module'].step()

            acc = self.train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss,)
            # Save a checkpoint
            if epoch % 5 == 4:
                acc = test(models, dataloaders, 'test')
                if best_acc < acc:
                    best_acc = acc
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict_backbone': models['backbone'].state_dict(),
                        'state_dict_module': models['module'].state_dict()
                    },
                    '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
                logging.error('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
        print('>> Finished.')
        logging.error(f'finish at epoch{epoch}')