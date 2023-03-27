import os
import sys
import torch
import numpy as np


class Tester:
    def __init__(self, args, loader, model, criterion, target_convert=None, data_aug=None):
        self.args = args
        self.loader = loader
        self.criterion = criterion
        self.task = args.task
        self.target_convert = target_convert
        if args.gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        model = model.to(self.device)
        self.model = model
        self.iter = 0
        self.aug = data_aug

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--gpu", type=bool, default=True)
        parser.add_argument("--n_gpu", type=int, default=1)
        port = (
                2 ** 15
                + 2 ** 14
                + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        )
        parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
        parser.add_argument("--print_every", type=int, default=50)
        parser.add_argument("--checkpoint_path", type=str, default="/nvme/fengbin/music2dance/fbwork/checkpoint_classifier/vqvae_2673.pt")
        parser.add_argument("--output_save_path", type=str, default="./outputs_posevq_2.12")
        parser.add_argument("--save_sample_num", type=int, default=100)
        return parser

    def test(self):
        loader = self.loader
        model = self.model
        criterion = self.criterion
        total_loss = {}
        total_loss["cnt"] = 0
        total_loss["acc"] = 0
        iter_num = 0

        model.eval()
        with torch.no_grad():
            for i, datas in enumerate(loader):
                iter_num += 1
                model.zero_grad()
                model_kwargs = {}
                targets = datas[0].to(self.device)
                beats = datas[1].to(self.device)
                styles = datas[2].to(self.device)
                contexts = datas[3].to(self.device)
                model_kwargs["beats"] = beats
                model_kwargs["styles"] = styles
                model_kwargs["contexts"] = contexts

                style_music = datas[4].to(self.device)
                same_style_music = datas[5].to(self.device)
                other_style_music = datas[6].to(self.device)
                model_kwargs["style_music"] = style_music
                model_kwargs["same_style_music"] = same_style_music
                model_kwargs["other_style_music"] = other_style_music

                if self.target_convert is not None:
                    targets = self.target_convert.run(targets)

                output = model(targets=targets, iter=self.iter, **model_kwargs)
                loss_dict, loss = criterion.compute(output, targets, **model_kwargs)

                for k, v in loss_dict.items():
                    if k not in total_loss.keys():
                        total_loss[k] = 0.
                    total_loss[k] += v
                total_loss["cnt"] += 1

        model.train()
        self.print_loss(iter_num, total_loss)

    def save(self, i, path):
        torch.save(self.model.state_dict(), f"{path}/vqvae_{str(i).zfill(3)}.pt")

    def print_loss(self, i, total_loss):
        loss_str = f""
        for k, v in total_loss.items():
            if k == "cnt":
                continue
            loss_str += f"{k}: {v/total_loss['cnt']:.5f};\t"

        print(
            f"valid\t iter:{i};\t" + loss_str
        )

    def load_from_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)