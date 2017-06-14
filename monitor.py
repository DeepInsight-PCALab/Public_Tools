import sys
import os
import os.path as osp
import re
import torch
import numpy as np
from collections import OrderedDict
from tabulate import tabulate


class ansi:
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog:
    '''
    Show log in color, easily indicate if loss is going down.
    Save log to a file simutaneously.
    '''
    def __init__(self, log_file=None, print_interval=1):
        self.first_iteration = True
        self.log_file = log_file
        self.print_interval = print_interval

    def __call__(self, info):
        if info['epoch'] % self.print_interval == 0:
            to_print = self.table(info, self.first_iteration)
            
            if self.first_iteration: self.first_iteration = False 
            
            print(to_print)
            sys.stdout.flush()
            
            if self.log_file is not None:
                for s in to_print.split('\n'):
                    os.system('echo {} >> {}'.format(self.decolorize(s), self.log_file))
    
    @staticmethod
    def table(info, first_iteration):
        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train loss', "{}{:.5f}{}".format(
                ansi.CYAN if info['train_loss_best'] else "",
                info['train_loss'],
                ansi.ENDC if info['train_loss_best'] else "",
                )),
            ('valid loss', "{}{:.5f}{}".format(
                ansi.GREEN if info['valid_loss_best'] else "",
                info['valid_loss'],
                ansi.ENDC if info['valid_loss_best'] else "",
                )),
            
            ])

        if 'valid_accuracy' in info:
            info_tabulate['valid acc'] = "{}{:.4f}{}".format(
                ansi.RED if info['valid_accuracy_best'] else "",
                info['valid_accuracy'],
                ansi.ENDC if info['valid_accuracy_best'] else "",
                )

        if 'current_lr' in info:
            info_tabulate['current_lr'] = info['current_lr']

        info_tabulate['dur'] = "{:.2f}s".format(info['dur'])

        ########Extend other infomation here#############

        tabulated = tabulate(
            [info_tabulate], headers="keys")

        out = ""
        if first_iteration:
            out = "\n".join(tabulated.split('\n', 2)[:2])
            out += "\n"

        out += tabulated.rsplit('\n', 1)[-1]
        return out

    @staticmethod
    def decolorize(string):
        color_pattern = r'\033\[\d+m'
        return re.sub(color_pattern, '', string)


class AutoSnapshot:
    """
    1. milestone: int, save model parameters every a specified interval
    2. lowerbound_trigger: float, save best parameters when accuracy is lager than it
    """
    def __init__(self, 
        path,
        milestone=200,
        lowerbound_trigger=0.99):
        self.path = path if path[-1] == '/' else path + '/'
        self.milestone = milestone
        self.lowerbound_trigger = lowerbound_trigger
        self.info_file = osp.join(self.path, 'snapshot_info.txt')
        self.first_iteration = True
        if not os.path.exists(self.path):
            os.makedirs(self.path)
       
    def __call__(self, model, info):
        if (info['epoch'] % self.milestone == 0 and info['epoch'] != 0) or \
           (info['valid_accuracy_best'] and info['valid_accuracy'] >= self.lowerbound_trigger):
            self.dump_model(model, info['epoch'])
            self.snap_record(info)

    def dump_model(self, model, epoch):
        torch.save(model.state_dict(), osp.join(self.path, 'epoch_{}.pth'.format(epoch)))

    def snap_record(self, info):
        model_info = PrintLog.table(info, self.first_iteration)
        if self.first_iteration: self.first_iteration = False 
        for s in model_info.split('\n'):
            os.system('echo {} >> {}'.format(PrintLog.decolorize(s), self.info_file))


class RememberBestWeights:
    def __init__(self, key='valid_loss'):
        self.key = key
        self.best_weights = None
        self.best_weights_loss = None
        self.best_weights_epoch = None

    def __call__(self, model, info):
        curr_loss = info[self.key]

        if info[self.key + '_best']:
            self.best_weights = model.state_dict()
            self.best_weights_loss = curr_loss
            self.best_weights_epoch = info['epoch']

    @staticmethod
    def store(best_weights, path, best_weights_epoch, best_weights_loss):
        filename = osp.join(path, 'BEST_epoch_{}_{}.pth'.format(best_weights_epoch, best_weights_loss))
        torch.save(self.best_weights, filename)



'''

sample usage:

#Define Class:
print_log = PrintLog(log_file = './log_sample.txt')
autosnap = AutoSnapshot('./snapshot_sample')
rememberbestweights = RememberBestWeights(key = 'valid_loss')

#After some iterations:
info = dict(
                epoch = epoch,
                train_loss = train_loss,
                valid_loss = valid_loss,
                train_loss_best = train_loss == best_train_loss,
                valid_loss_best = valid_loss == best_valid_loss,
                valid_accuracy = valid_acc,
                valid_accuracy_best = valid_acc == best_valid_acc,
                current_lr = curr_lr, 
                dur = time.time() - start_time
                )
print_log(info)
autosnap(model, info)
rememberbestweights(model, info)

'''