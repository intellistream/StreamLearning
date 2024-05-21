'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

import torch.nn.functional as F
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import time
# Torchvison
import torchvision.transforms as T
# import torchvision.models as models
from torchvision.datasets import CIFAR10


MARGIN = 1.0  # xi
WEIGHT = 1.0  # lambda

LR = 0.1
MILESTONES = [160]
EPOCHL = 120  # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4
device_global = 'cuda'


class LossNet(nn.Module):
    def __init__(self, num_channels=768, interm_dim=128):
        super(LossNet, self).__init__()

        self.FC1 = nn.Linear(num_channels, interm_dim)
        self.FC2 = nn.Linear(num_channels, interm_dim)
        self.FC3 = nn.Linear(num_channels, interm_dim)
        self.FC4 = nn.Linear(num_channels, interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = features[0].mean(1)
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = features[1].mean(1)
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = features[2].mean(1)
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = features[3].mean(1)
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


def get_uncertainty(model, unlabeled_loader):
    orig_mode = model.training
    model.eval()
    device = 'cuda'
    uncertainty = torch.tensor([]).to(device)

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.to(device)
            pred_loss = model(inputs, learnloss=True)
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    model.train(orig_mode)

    return uncertainty.cpu()

def learnloss_selection(x, y, model, size):
    new_dataset = TensorDataset(x, y)
    data_loader_train = DataLoader(new_dataset, batch_size=x.shape[0], shuffle=False)
    uncertainty = get_uncertainty(model, data_loader_train)

    # Index in ascending order
    indices = np.argsort(uncertainty)[:size]
    return indices
