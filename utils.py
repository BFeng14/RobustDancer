import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from scheduler import CycleScheduler

def create_dataloader(dataset, split="train"):
    is_eval = (split == "val" or split == "test")
    return DataLoader(dataset,
                      batch_size=dataset.arg.batch_size if not is_eval else dataset.arg.val_batch_size,
                      shuffle=not is_eval,
                      # collate_fn=meta_collate_fn(dataset.opt.pad_batches,dataset.opt.model),
                      # pin_memory=True,
                      drop_last=True,
                      num_workers=dataset.arg.workers)


def create_optimizer(args, model, weight_decay=0):
    optimizer = None
    if args.optim == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=weight_decay)
    return optimizer


def create_scheduler(args, optimizer, loader):
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )
    return scheduler


import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import torch.nn.init as init
import math

def init_rnn(x, type='uniform'):
    for layer in x._all_weights:
        for w in layer:
            if 'weight' in w:
                if type == 'xavier':
                    init.xavier_normal_(getattr(x, w))
                elif type == 'uniform':
                    stdv = 1.0 / math.sqrt(x.hidden_size)
                    init.uniform_(getattr(x, w), -stdv, stdv)
                elif type == 'normal':
                    stdv = 1.0 / math.sqrt(x.hidden_size)
                    init.normal_(getattr(x, w), .0, stdv)
                else:
                    raise ValueError