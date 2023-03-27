import argparse
import os
import numpy as np
from pathlib import Path
from utils import *
from models_final.condition_transformer_rawaistpp_style_hist_unpaired_v2 import DanceModel
from dataset.final_dataset import Pose3dMusicDataset


def inference(args, model, name_all, pose, music, beat):
    musics = torch.tensor(music).float().cuda()
    targets = torch.tensor(pose).float().cuda()
    beats = torch.tensor(beat).long().cuda()
    styles = targets[:, :40, :]
    result = model.generate_old(targets, music=musics, style=styles, beats=beats).squeeze(0)
    result = result.cpu().detach().numpy()
    for i, name in enumerate(name_all):
        np.save(os.path.join(args.save_path, "result_{}".format(name)), result[i])
    return result


def reshape_feature(feature, diff_beat):
    feature_len = feature.shape[0]
    dim_num = feature.shape[1]
    target_len = int((feature_len / diff_beat) * 10)
    x = [float(x*(feature_len-1))/float(target_len-1) for x in range(target_len)]
    xp = [x for x in range(feature_len)]
    res = []
    for i in range(dim_num):
        fp = feature[:, i]
        res.append(np.interp(x, xp, fp))
    res = np.array(res)
    res = np.transpose(res)
    return res, target_len


def read_data(data_name):
    music_data_path = "/nvme/fengbin/music2dance/music2dance_data/aistpp_feature_mirror/audio_feature_20"
    # music_mod = "audio_feats_scaled_20"
    audio_names = data_name.split("_")[-2]
    music_data = np.load(os.path.join(music_data_path, audio_names + ".npy"))

    # data_name_mirror = data_name+"mirror"
    pose_data_path = "/nvme/fengbin/music2dance/music2dance_data/aistpp_feature_mirror/motion_feature_20_beat_3"
    # pose_mod = "expmap_cr_scaled_20"
    pose_data = np.load(os.path.join(pose_data_path, data_name + ".npy"))
    beats_data = music_data[:, -1]
    dist = 0

    for i in range(music_data.shape[0] - 1, -1, -1):
        if beats_data[i] > 0:
            dist = 0
        beats_data[i] = min(dist, 40)
        dist += 1
    music_data = music_data[:, 1:33]
    pose_data = pose_data[:, :-1]
    # pose_data_mirror = np.load(os.path.join(pose_data_path, data_name_mirror + ".npy"))

    return np.expand_dims(music_data[:20*20+40], axis=0), \
           np.expand_dims(pose_data[:40], axis=0), \
           np.expand_dims(beats_data[:20*20+80], axis=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # task args
    parser.add_argument("--model", type=str, default="vqvae")
    parser.add_argument("--dataset", type=str, default="pose")
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--save_path", type=str, default="./np_result_style_hist_v2_new_new_final_130900")

    # GCN model args
    parser = DanceModel.modify_commandline_options(parser)
    # criterion args
    # parser = SeqGenCrit.modify_commandline_options(parser)
    # dataset args
    parser = Pose3dMusicDataset.modify_commandline_options(parser)
    # model path
    parser.add_argument("--checkpoint_path", type=str, default="/data1/fengbin/checkpoints/checkpoint_style_v2_new_new_final/vqvae_130900.pt")
    args = parser.parse_args()
    model = DanceModel(args)
    model.load_state_dict(torch.load(args.checkpoint_path), strict=False)
    model = model.cuda()
    model.eval()

    split = "test"

    data_path = Path("/nvme/fengbin/music2dance/music2dance_data/aist_plusplus_final/splits")
    ignore_files = [x.strip() for x in open(data_path.joinpath("ignore_list.txt"), "r").readlines()]

    base_filenames = []

    if split == "train":
        base_filenames += [x.strip() for x in
                           open(data_path.joinpath("crossmodal_train.txt"), "r").readlines()]
    else:
        base_filenames += [x.strip() for x in
                           open(data_path.joinpath("crossmodal_test.txt"), "r").readlines()]
        base_filenames += [x.strip() for x in
                           open(data_path.joinpath("crossmodal_val.txt"), "r").readlines()]
    # temp_base_filenames = ["aistpp_" + x for x in base_filenames if x not in ignore_files]

    # np.random.seed(1111)
    # data_name = np.random.choice(base_filenames, size=6,
    #                              replace=False).tolist()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    from tqdm import tqdm

    music_all = []
    pose_all = []
    beat_all = []
    name_all = []
    for name in tqdm(base_filenames):
        music, pose, beat = read_data(name)
        music_all.append(music)
        pose_all.append(pose)
        beat_all.append(beat)
        name_all.append(name)
    inference(args, model, name_all[:20],
                       np.concatenate(pose_all[:20], axis=0),
                       np.concatenate(music_all[:20], axis=0),
                       np.concatenate(beat_all[:20], axis=0))
    inference(args, model, name_all[20:],
              np.concatenate(pose_all[20:], axis=0),
              np.concatenate(music_all[20:], axis=0),
              np.concatenate(beat_all[20:], axis=0))
