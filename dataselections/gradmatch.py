from cords.cords.selectionstrategies.SL.gradmatchstrategy import GradMatchStrategy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from timm.models import create_model
import models

class Gradmatch_Selection:
    def __init__(self, trainloader, config):
        valloader = trainloader
        num_classes = config['num_classes']
        eta = 0.001
        device = 'cuda'
        linear_layer = False
        loss = nn.CrossEntropyLoss(reduction='mean')
        selection_type = 'PerBatch'
        logger = print
        valid = False

        model = models.__dict__[config['model_type']].__dict__[config['model_name']](out_dim=config['out_dim'],
                                                                                     args=config['args'],
                                                                                     prompt_flag='coda',
                                                                                     prompt_param=config['prompt_param']).to(device)
        self.gradmatch = GradMatchStrategy(trainloader, valloader, model, loss, eta,
                                           device, num_classes, linear_layer, selection_type, logger, valid)

    def select(self, x, y, valid_out_dim, last_valid_out_dim,  size, model_params):
        self.set_new_trainloader(x, y)
        idxs, _ = self.gradmatch.select(valid_out_dim=valid_out_dim, last_valid_out_dim=last_valid_out_dim,
                                        budget=size, model_params=model_params)
        return idxs

    def set_new_trainloader(self, x, y):
        new_dataset = TensorDataset(x, y)
        self.gradmatch.trainloader = DataLoader(new_dataset, batch_size=x.shape[0], shuffle=False)
        self.gradmatch.N_trn = len(self.gradmatch.trainloader.sampler)
