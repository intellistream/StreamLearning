from __future__ import print_function
import torch
import models
from .default import NormalNN
from dataselections.learnloss import LossPredLoss, MARGIN, LR
import time
from utils.schedulers import CosineSchedule
from models.zoo import CodaPrompt, DualPrompt, L2P
from buffer.buffer import DynamicBuffer, Buffer
from buffer.buffer_utils import maybe_cuda

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.args = learner_config['args']
        super(Prompt, self).__init__(learner_config)
        update_method = ['streamprompt', 'random', 'aser', 'camel', 'gss', 'summarize']
        if self.args.mem_size > 0 and self.args.update_method in update_method:
            if learner_config['args'].update_method == 'summarize':
                buffer = DynamicBuffer
            else:
                buffer = Buffer
            self.buffer = buffer(model=self.model, config=learner_config)
        else:
            self.buffer = None
        self.buffer_time = 0

    def update_prompt_model(self, inputs, targets, all_inputs, all_targets, iteration):
        if self.buffer is None:
            # logits
            logits, prompt_loss, pred_loss = self.model(inputs, train=True)
            logits = logits[:, :self.valid_out_dim]
            # ce with heuristic
            logits[:, :self.last_valid_out_dim] = -float('inf') # only masking the inputs except from the buffer data
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            # total_loss_ori = self.criterion(logits, targets.long(), dw_cls)
            ori_loss = self.criterion_fn(logits, targets.long())
            total_loss = (ori_loss * dw_cls).mean()
            # pred loss
            if pred_loss is not None:
                pred_loss = pred_loss.view(pred_loss.size(0))
                module_loss = LossPredLoss(pred_loss, ori_loss, margin=MARGIN)
                total_loss = total_loss + prompt_loss.sum() + module_loss
            else:
                # ce loss
                total_loss = total_loss + prompt_loss.sum()
            # step
            self.optimizer.zero_grad()
            if self.model.learnloss:
                self.lossnet_optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if self.model.learnloss:
                self.lossnet_optimizer.step()
        else:
            mem_x, mem_y = self.buffer.retrieve(x=all_inputs, y=all_targets)
            if self.args.update_method not in ['aser', 'summarize']:
                logits, prompt_loss, _ = self.model(inputs, train=True)
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                total_loss = self.criterion(logits, targets.long(), dw_cls)
                total_loss = total_loss + prompt_loss.sum()
                self.optimizer.zero_grad()
                total_loss.backward()
                # mem update
                if mem_x.size(0) > 0:
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    mem_y = maybe_cuda(mem_y, self.cuda)
                    dw_cls = self.dw_k[-1 * torch.ones(mem_y.size()).long()]
                    mem_logits, prompt_loss, _ = self.model(mem_x.float(), train=True)
                    total_loss = self.criterion(mem_logits, mem_y.long(), dw_cls)
                    total_loss = total_loss + prompt_loss.sum()
                    total_loss.backward()
                    self.optimizer.step()
                else:
                    self.optimizer.step()
            if self.args.update_method == 'summarize':
                if mem_x.size(0) > 0:
                    batch_size = inputs.size(0)
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    mem_y = maybe_cuda(mem_y, self.cuda)
                    combined_batch = torch.cat((mem_x, inputs))
                    combined_labels = torch.cat((mem_y, targets))
                    dw_cls = self.dw_k[-1 * torch.ones(combined_labels.size()).long()]
                    combined_batch_aug = self.buffer.transform(combined_batch)
                    feat, prompt_loss, _ = self.model(combined_batch, train=True)
                    feat_aug, prompt_loss_aug, _ = self.model(combined_batch_aug, train=True)
                    # features = torch.cat([feat.unsqueeze(1), feat_aug.unsqueeze(1)], dim=1)
                    features = feat + feat_aug
                    total_loss = self.criterion(features, combined_labels.long(), dw_cls)
                    total_loss = total_loss + prompt_loss.sum() + prompt_loss_aug.sum()
                    logits = feat[-batch_size:]
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                else:
                    logits, prompt_loss, _ = self.model(inputs, train=True)
                    dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                    total_loss = self.criterion(logits, targets.long(), dw_cls)
                    total_loss = total_loss + prompt_loss.sum()
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
            if self.args.update_method == 'aser':
                if mem_x.size(0) > 0:
                    # opt update
                    batch_size = inputs.size(0)
                    combined_batch = torch.cat((mem_x, inputs))
                    combined_labels = torch.cat((mem_y, targets))
                    dw_cls = self.dw_k[-1 * torch.ones(combined_labels.size()).long()]
                    combined_logits, prompt_loss, _ = self.model(combined_batch, train=True)
                    total_loss = self.criterion(combined_logits, combined_labels.long(), dw_cls)
                    total_loss = total_loss + prompt_loss.sum()
                    logits = combined_logits[-batch_size:]
                else:
                    logits, prompt_loss, _ = self.model(inputs, train=True)
                    dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                    total_loss = self.criterion(logits, targets.long(), dw_cls)
                    total_loss = total_loss + prompt_loss.sum()
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # update mem
            if self.args.update_method == 'summarize':
                # update memory
                self.buffer.aff_x.append(all_inputs)
                self.buffer.aff_y.append(all_targets)
                if len(self.buffer.aff_x) > self.buffer.update_method.params.queue_size:
                    self.buffer.aff_x.pop(0)
                    self.buffer.aff_y.pop(0)
                start = time.time()
                self.buffer.update(all_inputs, all_targets, aff_x=self.buffer.aff_x, aff_y=self.buffer.aff_y,
                                   update_index=iteration, transform=self.buffer.transform)
                self.buffer_time += time.time() - start
            else:
                start = time.time()
                self.buffer.update(all_inputs.cuda(), all_targets.cuda())
                self.buffer_time += time.time() - start

        return total_loss.detach(), logits


    # sets model optimizers
    def init_optimizer(self):
        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            if self.args.prompt_attune == 1:
                # params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
                params_to_opt = list(self.model.last.parameters())

                params_prompt = []
                # add the moe layer's params
                for name, param in self.model.prompt.named_parameters():
                    if not any(f"moe.moe.experts.{i}" in name for i in range(self.model.prompt.moe.num_old_experts)):
                        params_prompt.append(param)

                params_to_opt += params_prompt
            else:
                params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())

            # add LossNet params
            if self.model.learnloss:
                print(f'Initialize Optimizer for LossNet')
                params_lossnet = []
                for name, param in self.model.prompt.named_parameters():
                    params_lossnet.append(param)

        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.model.learnloss:
            lossnet_optimizer_arg = {'params':params_lossnet,
                                     'lr': 0.0001,
                                     # 'lr': LR,
                                     'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        if self.model.learnloss:
            self.lossnet_optimizer = torch.optim.__dict__[self.config['optimizer']](**lossnet_optimizer_arg)

        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)
        else:
            self.scheduler = None

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim,
                                                                               args=cfg['args'],
                                                                               prompt_flag='coda',
                                                                               prompt_param=self.prompt_param,
                                                                               learnloss=(cfg['selection_method']=='learnloss'))
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self, prompt=None, augment_model_flag=False):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim,
                                                                               args=cfg['args'],
                                                                               prompt_flag='dual',
                                                                               prompt_param=self.prompt_param,
                                                                               learnloss=(cfg['selection_method']=='learnloss'))
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self, prompt=None, augment_model_flag=False):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim,
                                                                               args=cfg['args'],
                                                                               prompt_flag='l2p',
                                                                               prompt_param=self.prompt_param,
                                                                               learnloss=(cfg['selection_method']=='learnloss'))
        return model