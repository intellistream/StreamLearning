from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
import torch
import numpy as np
import yaml
import csv
import random
from trainer import Trainer
import scipy.stats as stats
from scipy.stats import sem

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    # Data selection
    parser.add_argument('--selection_method', type=str, default='None', help='Different data selection')
    parser.add_argument('--selection_ratio', type=float, default=0.5, help='Data selection ratio')
    parser.add_argument('--prompt_attune', type=int, default=1, help='Apply prompt attunement or not')
    # Buffer update
    parser.add_argument('--mem_size', type=int, default=102, help='Rehearsal memory size')
    parser.add_argument('--update_method', type=str, default='camel', help='Memory update strategy')
    parser.add_argument('--gss_batch_size', type=int, default=10, help='GSS batch size')
    parser.add_argument('--eps_mem_batch', type=int, default=10, help='Episode memory per batch (default: %(default)s)')
    # Fast stream
    parser.add_argument('--traintime_limit', type=int, default=100, help='The limitation of training time for fast stream')
    parser.add_argument('--skip_batch', type=int, default=0, help='Skip batch for fast stream')
    parser.add_argument('--file_name', type=str, default="results_log/results.csv", help='The path to store results')
    # Pretrained model
    parser.add_argument('--ptm', type=str, default="None", help='The path to store results')
    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/cifar-100/10-task/coda-p",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='prompt', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='CODAPrompt', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=1, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[100, 8, 0.0],
                         help="e prompt pool size, e prompt length, g prompt length")

    # Config Arg
    parser.add_argument('--config', type=str, default="configs/stream51_prompt.yaml", help="yaml experiment config input")
    # add_sparse_args(parser)
    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    return argparse.Namespace(**config)

def analyze_results(results, indices=None, top_5=False):
    if top_5:
        if indices is None:
            indices = np.argsort(results)[-5:]
            indices = indices[::-1]
            results = results[indices]
        else:
            results = results[indices]

    mean = np.mean(results)
    t_coef = stats.t.ppf((1 + 0.95) / 2, len(results) - 1)
    margin_of_error = t_coef * sem(results)

    return mean, margin_of_error, indices

def print_save_res(acc_res, fgt_res, train_time_res, buffer_time_res, file_name, top_5=False):
    acc_mean, acc_dif, indices = analyze_results(acc_res, None, top_5)
    fgt_mean, fgt_dif, _ = analyze_results(fgt_res, indices, top_5)
    train_time_mean, train_time_dif, _ = analyze_results(train_time_res, indices, top_5)
    buffer_time_mean, buffer_time_dif, _ = analyze_results(buffer_time_res, indices, top_5)
    res_str = f'**[{args.selection_method}] | [{args.update_method}] | For all {args.repeat} runs:'
    res_str += f'\nAcc:{acc_mean:.2f}+-{acc_dif:.2f}'
    res_str += f'\tForgetting:{fgt_mean:.2f}+-{fgt_dif:.2f}'
    res_str += f'\tTraining Time:{train_time_mean:.2f}+-{train_time_dif:.2f}'
    res_str += f'\tBuffer Update Time:{buffer_time_mean:.2f}+-{buffer_time_dif:.2f}'
    print(res_str)

    if not top_5:
        row_data = [f'{args.selection_method}--All {args.repeat} runs',
                    f"{train_time_mean:.2f}\pm{train_time_dif:.2f}",
                    f"{acc_mean:.2f}\pm{acc_dif:.2f}",
                    f"{fgt_mean:.2f}\pm{fgt_dif:.2f}",
                    f"{buffer_time_mean:.2f}\pm{buffer_time_dif:.2f}"]

    else:
        row_data = [f'{args.selection_method}--Top 5 runs',
                    f"{train_time_mean:.2f}\pm{train_time_dif:.2f}",
                    f"{acc_mean:.2f}\pm{acc_dif:.2f}",
                    f"{fgt_mean:.2f}\pm{fgt_dif:.2f}",
                    f"{buffer_time_mean:.2f}\pm{buffer_time_dif:.2f}"]

    # file_path = file_name.replace('txt', 'csv')
    print(f'Save in: {file_name}')
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    file_name = args.file_name
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    print('************************************')
    print(args)

    acc_res = np.zeros(args.repeat)
    fgt_res = np.zeros(args.repeat)
    bwt_res = np.zeros(args.repeat)
    train_time_res = np.zeros(args.repeat)
    buffer_time_res = np.zeros(args.repeat)
    for r in range(args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')
        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # set up a trainer
        trainer = Trainer(args, seed, r)

        # train model
        trainer.train([r, args.repeat])

        diagonal = np.diag(trainer.acc_matrix)
        forgetting = np.mean((np.max(trainer.acc_matrix, axis=0) - trainer.acc_matrix[-1, :])[:(trainer.max_task - 1)])
        backward = np.mean((trainer.acc_matrix[-1, :] - diagonal)[:(trainer.max_task - 1)])
        r_res = f'In run {r+1}:\n'
        r_res += f'The task labels is: {trainer.tasks}\n'
        r_res += f'Acc: {trainer.acc_matrix[-1, :].mean():.2f}\t' \
                 f' Forgetting: {forgetting:.2f}\t' \
                 f' Backward: {backward:.2f}\t' \
                 f' Training time: {trainer.train_time.sum():.2f}\t' \
                 f' Buffer update time: {trainer.learner.buffer_time:.2f}\n'
        print(r_res)

        headers = [" ", "Train Time", "Acc", "Forgetting", "Buffer Update Time", "Task Labels"]
        row_data = [f"{r+1} runs", f"{trainer.train_time.sum():.2f}",
                    f"{trainer.acc_matrix[-1, :].mean():.2f}",
                    f"{forgetting:.2f}",
                    f"{trainer.learner.buffer_time:.2f}",
                    f"{trainer.tasks}"]
        print(f'Save in: {file_name}')
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if r == 0:
                writer.writerow(headers)
            writer.writerow(row_data)

        acc_res[r] = trainer.acc_matrix[-1, :].mean()
        fgt_res[r] = forgetting
        bwt_res[r] = backward
        train_time_res[r] = trainer.train_time.sum()
        buffer_time_res[r] = trainer.learner.buffer_time

    print_save_res(acc_res, fgt_res, train_time_res, buffer_time_res, file_name, top_5=False)


