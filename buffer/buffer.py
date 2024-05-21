import torch
import numpy as np
import copy
from collections import defaultdict, namedtuple
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from buffer.random_retrieve import Random_retrieve
from buffer.buffer_utils import maybe_cuda
from buffer.update_method import update_methods


class Buffer(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        Args = namedtuple('Args', ['cuda', 'num_classes_per_task', 'images_per_class', 'mem_size',
                                   'eps_mem_batch', 'gss_batch_size', 'dataset', 'update'])
        params = Args(cuda=True, num_classes_per_task=10, images_per_class=10, mem_size=config['args'].mem_size,
                      eps_mem_batch=config['args'].eps_mem_batch, gss_batch_size=config['args'].gss_batch_size,
                      dataset=config['args'].dataset, update=config['args'].update_method)

        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"
        self.num_classes_per_task = self.params.num_classes_per_task
        self.num_classes = 0

        # define buffer
        buffer_size = params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = [3, 224, 224]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        self.labeldict = defaultdict(list)
        self.labelsize = params.images_per_class
        self.avail_indices = list(np.arange(buffer_size))

        # define update and retrieve method
        self.update_method = update_methods[params.update](params)
        self.retrieve_method = Random_retrieve(params)


    def update(self, x, y, **kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)

    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def new_task(self, labels, **kwargs):
        self.num_classes += len(labels)
        # self.update_method.new_task(self.num_classes, labels)



class DynamicBuffer(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        Args = namedtuple('Args', ['cuda', 'images_per_class', 'eps_mem_batch', 'mem_size', 'update'])
        params = Args(cuda=True, images_per_class=config['args'].mem_size//config['num_classes'],
                      eps_mem_batch=config['args'].eps_mem_batch,
                      mem_size=config['args'].mem_size, update=config['args'].update_method)
        self.config = config
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2))
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"
        self.images_per_class = self.params.images_per_class
        self.num_classes = 0

        # self.transform = nn.Sequential(RandomResizedCrop(size=(224, 224), scale=(0.2, 1.)),
        #                                RandomHorizontalFlip(),
        #                                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        #                                RandomGrayscale(p=0.2))
        # define buffer
        buffer_size = params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = [3, 224, 224]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_img_rep', copy.deepcopy(buffer_img))
        self.register_buffer('buffer_label', buffer_label)
        self.condense_dict = defaultdict(list)
        self.labelsize = params.images_per_class
        self.avail_indices = list(np.arange(buffer_size))

        # define update and retrieve method
        self.update_method = update_methods[params.update](config)
        self.retrieve_method = Random_retrieve(params)

    def update(self, x, y, **kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)

    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def new_task(self, labels, **kwargs):
        self.aff_x = []
        self.aff_y = []
        self.num_classes += len(labels)
        self.update_method.new_task(self.num_classes, labels)
