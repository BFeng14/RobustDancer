import torch
from torch import nn

class Crit:
    def __init__(self, args):
        self.lossfun = nn.MSELoss()

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def compute(self, output, target, **kwargs):
        loss_dict = {}
        for k, v in output.items():
            loss_dict[k] = v.item()
        loss = output["loss"]
        return loss_dict, loss
