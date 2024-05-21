from cords.cords.selectionstrategies.SL.craigstrategy import CRAIGStrategy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from timm.models import create_model
import models

class Craig_Selection:
    def __init__(self, trainloader, config):

        valloader = trainloader
        num_classes = config['num_classes']
        device = 'cuda'
        linear_layer = False
        loss = nn.CrossEntropyLoss(reduction='mean')
        selection_type = 'PerBatch'
        logger = print
        if_convex = False
        optimizer = 'lazy'

        model = models.__dict__[config['model_type']].__dict__[config['model_name']](out_dim=config['out_dim'],
                                                                                     args=config['args'],
                                                                                     prompt_flag='coda',
                                                                                     prompt_param=config['prompt_param']).to(device)
        self.craig = CRAIGStrategy(trainloader, valloader, model, loss, device, num_classes,
                                   linear_layer, if_convex, selection_type, logger, optimizer)

    def select(self, x, y, valid_out_dim, last_valid_out_dim, size, model_params):
        self.set_new_trainloader(x, y)
        idxs, _ = self.craig.select(valid_out_dim=valid_out_dim, last_valid_out_dim=last_valid_out_dim,
                                    budget=size, model_params=model_params)
        return idxs

    def set_new_trainloader(self, x, y):
        new_dataset = TensorDataset(x, y)
        self.craig.trainloader = DataLoader(new_dataset, batch_size=x.shape[0], shuffle=False)
        self.craig.N_trn = len(self.craig.trainloader.sampler)
