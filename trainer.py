import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners

class Trainer:

    def __init__(self, args, seed, run):

        # process inputs
        self.seed = seed
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1

        self.dataset = args.dataset
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'Tiny_ImageNet':
            Dataset = dataloaders.iTINY_IMAGENET
            num_classes = 200
            self.dataset_size = [64,64,3]
        elif args.dataset == 'Stream51':
            Dataset = dataloaders.iSTREAM51
            num_classes = 51
            self.dataset_size = [64,64,3]
        elif args.dataset == 'Clear10':
            Dataset = dataloaders.iCLEAR10
            num_classes = 11
            self.dataset_size = [64,64,3]
        elif args.dataset == 'Clear100':
            Dataset = dataloaders.iCLEAR100
            num_classes = 100
            self.dataset_size = [64, 64, 3]
        elif args.dataset == 'Core50':
            Dataset = dataloaders.iCORE50
            num_classes = 50
            self.dataset_size = [128,128,3]
        elif args.dataset == 'Mini_ImageNet':
            Dataset = dataloaders.iMINI_IMAGENET
            num_classes = 100
            self.dataset_size = [84, 84, 3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 345
            self.dataset_size = [224,224,3]
            self.top_k = 1
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        # if args.rand_split:
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            np.random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        self.acc_matrix = np.zeros((self.max_task, self.max_task))
        self.forgetting = np.zeros(self.max_task - 1)
        self.train_time = np.zeros(self.max_task)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, cur_run=run,
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    download_flag=False, transform=test_transform, cur_run=run,
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0
        traintime = {
            'Stream51': {'None': 1755.66, 'random': 1883.75, 'camel': 9546.85, 'streamprompt': 2026.92},
            'Core50': {'None': 1395.67, 'random': 1497.5, 'camel': 7589.33, 'streamprompt': 1617.3},
            'Clear10': {'None': 384.45, 'random': 412.5, 'camel': 2090.55, 'streamprompt': 445.5},
            'Clear100': {'None': 1280.34, 'random': 1373.75, 'camel': 6962.17, 'streamprompt': 1483.65}
        }
        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                                'task_labels': self.train_dataset.tasks if self.dataset == 'Core50' else self.tasks,
                                'lr': args.lr,
                                'learner_name': args.learner_name,
                                'debug_mode': args.debug_mode == 1,
                                'momentum': args.momentum,
                                'selection_method': args.selection_method,
                                'selection_ratio': args.selection_ratio,
                                'traintime': traintime[args.dataset],
                                'skip_batch': args.skip_batch == 1,
                                'args': args,
                                'weight_decay': args.weight_decay,
                                'schedule': args.schedule,
                                'schedule_type': args.schedule_type,
                                'model_type': args.model_type,
                                'model_name': args.model_name,
                                'optimizer': args.optimizer,
                                'gpuid': args.gpuid,
                                'temp': args.temp,
                                'out_dim': num_classes,
                                'overwrite': args.overwrite == 1,
                                'DW': args.DW,
                                'batch_size': args.batch_size,
                                'upper_bound_flag': args.upper_bound_flag,
                                'tasks': self.tasks_logits,
                                'n_tasks': self.max_task,
                                'top_k': self.top_k,
                                'prompt_param':[self.num_tasks, args.prompt_param]
                                }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, run):

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print(f'====================== Run {run[0]+1}/{run[1]} | Task {train_name}/{self.max_task} ======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            # self.train_dataset.append_coreset(only=False)

            # load dataloader
            # if self.dataset in ['Stream51', 'Core50']:
            if self.dataset in ['Stream51', 'Core50', 'Clear10', 'Clear100']:
                train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                          drop_last=True, num_workers=int(self.workers))
            else:
                train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                            drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader,
                                                      task_id=i, config=self.learner_config)

            # evaluate acc
            acc_table = []
            self.reset_cluster_labels = True

            if i in list(range(self.max_task)):
                for j in range(i+1):
                    acc = self.task_eval(j)
                    acc_table.append(acc)
                    self.acc_matrix[i, j] = acc
                if i > 0:
                    self.forgetting[i-1] = np.mean((np.max(self.acc_matrix, axis=0) - self.acc_matrix[i, :])[:i])

            print(f'=> Mean Acc: {self.acc_matrix[i, :i+1].mean():.4f} | Train time:{avg_train_time:.4f}')
            self.train_time[i] = avg_train_time


    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics