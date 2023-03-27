import argparse
from criterion.criterion_attn import Crit
from utils import *
from trainer import Trainer
from tester import Tester
from dataset.final_dataset import Pose3dMusicDataset
from models_final.condition_transformer_rawaistpp_style_hist_unpaired_v2 import DanceModel
from utils import str2bool

# CUDA_VISIBLE_DEVICES=0 nohup python -u train_seq_model_transformer_rawdata_style_hist_unpaired.py  --save_path /data1/fengbin/checkpoints/checkpoint_style_hist_aaai_final > /data1/fengbin/checkpoints/checkpoint_style_hist_aaai_final/train_log 2>&1 &


def main(args):
    train_dataset = Pose3dMusicDataset(args, split="train")
    train_loader = create_dataloader(train_dataset)
    test_dataset = Pose3dMusicDataset(args, split="test")
    test_loader = create_dataloader(test_dataset)
    model = DanceModel(args)
    optimizer = create_optimizer(args, model)
    criterion = Crit(args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 100], gamma=0.1, last_epoch=-1)
    tester = Tester(args, test_loader, model, criterion)
    trainer = Trainer(args, train_loader, model, optimizer, criterion, tester, scheduler=scheduler)
    trainer.load_from_checkpoint()
    trainer.start_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # task args
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--dataset", type=str, default="pose")
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--use_codebook_weight", type=str2bool, default=False)
    # optimizer args
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sched", type=str, default="cycle")
    # train args
    parser = Trainer.modify_commandline_options(parser)
    # dataset args
    parser = Pose3dMusicDataset.modify_commandline_options(parser)
    # GCN model args
    parser = DanceModel.modify_commandline_options(parser)
    # criterion args
    parser = Crit.modify_commandline_options(parser)

    args = parser.parse_args()
    print(args)
    main(args)