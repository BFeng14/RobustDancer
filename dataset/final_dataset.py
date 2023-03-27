import sys
sys.path.append("..")
# from LTD_utils import data_utils
from pathlib import Path
import numpy as np
import torch
import random
from dataset.base_dataset import BaseDataset
import scipy.interpolate as spi
from random import choice
from utils import str2bool
import pickle
import copy


class Pose3dMusicDataset(BaseDataset):

    def __init__(self, arg, split="train"):
        super().__init__()
        self.arg = arg
        data_path = Path(arg.data_dir)
        split_path = Path(arg.split_dir)
        music_data_path = Path(arg.music_data_dir)
        # foot_data_path = Path(arg.foot_data_dir)
        self.split = split

        random.seed(1111)

        if not data_path.is_dir():
            raise ValueError('Invalid directory:' + arg.data_dir)

        ignore_files = [x.strip() for x in open(split_path.joinpath("ignore_list.txt"), "r").readlines()]

        base_filenames = []

        if split == "train":
            base_filenames += [x.strip() for x in
                               open(split_path.joinpath("crossmodal_train.txt"), "r").readlines()]
        else:
            base_filenames += [x.strip() for x in
                               open(split_path.joinpath("crossmodal_test.txt"), "r").readlines()]
            base_filenames += [x.strip() for x in
                               open(split_path.joinpath("crossmodal_val.txt"), "r").readlines()]

        base_filenames = [x for x in base_filenames if x not in ignore_files]
        # temp_base_filenames = ["aistpp_" + x for x in base_filenames if x not in ignore_files]
        if self.arg.mirror_aug:
            base_filenames += [x+"mirror" for x in base_filenames]

        np.random.seed(1111)
        if arg.num_train_samples > 0 and arg.num_train_samples < len(base_filenames):
            base_filenames = np.random.choice(base_filenames, size=arg.num_train_samples,
                                                   replace=False).tolist()
        self.base_filenames = []
        self.pose_features = {}
        self.music_features = {}
        self.avg_beats = {}
        self.style2music = {}
        self.total_frames = 0
        self.frame_cum_sums = []
        self.input_length = arg.total_len

        frame_sum = 0

        # Get the list of files containing features (in numpy format for now), and populate the dictionaries of input and output features (separated by modality)
        for base_filename in base_filenames:
            ignore_file = False
            pose_feature_file = data_path.joinpath(base_filename + ".npy")
            audio_names = base_filename.split("_")[-2]
            if audio_names not in self.music_features.keys():
                music_feature_file = music_data_path.joinpath(audio_names + ".npy")
                music_features = np.load(music_feature_file)[:, 1:33]
                self.music_features[audio_names] = music_features
                style_name = audio_names[1:3]
                if style_name not in self.style2music.keys():
                    self.style2music[style_name] = set()
                self.style2music[style_name].add(audio_names)

            try:
                pose_features = np.load(pose_feature_file)
                frame_sum += pose_features.shape[0]
                pose_length = pose_features.shape[0]
                if pose_length < self.input_length:
                    print("Smol sequence " + base_filename + "." + arg.pose_mod + "; ignoring..")
                    ignore_file = True
            except Exception as e:
                print(e)
                continue

            if ignore_file: continue
            self.pose_features[base_filename] = pose_features
            self.base_filenames.append(base_filename)

        for k,v in self.style2music.items():
            self.style2music[k] = list(v)
        print("sequences added: " + str(len(self.base_filenames)))
        print(frame_sum)

    def process_beats(self, music_beats):
        beats_index = []
        for i in range(music_beats.shape[0]):
            if music_beats[i] > 0:
                beats_index.append(i)

        beats_offsets = [beats_index[i+1] - beats_index[i] for i in range(len(beats_index)-1)]
        avg_beats = sum(beats_offsets) // len(beats_offsets)
        return avg_beats

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--output_n", type=int, default=7)
        parser.add_argument("--input_n", type=int, default=40)
        parser.add_argument("--history_n", type=int, default=40)
        parser.add_argument("--total_len", type=int, default=80)
        parser.add_argument('--split_dir', type=str,
                            default="path_to_anno_dir/aist_plusplus_final/splits")
        parser.add_argument('--data_dir', type=str,
                            default="path_to_aistpp_feature_mirror/motion_feature_20_beat_3")
        parser.add_argument('--music_data_dir', type=str,
                            default="path_to_aistpp_feature_mirror/audio_feature_20")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--val_batch_size", type=int, default=128)
        parser.add_argument("--num_train_samples", type=int, default=0)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--augment_rate", type=float, default=0.0)
        parser.add_argument("--joint_aug_rate", type=float, default=0.6)
        parser.add_argument("--mirror_aug", type=str2bool, default=True)
        parser.add_argument("--offset_aug", type=str2bool, default=False)
        parser.add_argument("--dataset_repeat", type=int, default=80)
        return parser

    def name(self):
        return "Music2PoseDataset"

    def __getitem__(self, item):
        idx = item % len(self.base_filenames)
        base_filename = self.base_filenames[idx]

        pose_features = self.pose_features[base_filename]

        audio_names = base_filename.split("_")[-2]
        style_names = audio_names[1:3]
        all_style = set(self.style2music.keys())
        all_style.remove(style_names)
        other_style = choice(list(all_style))

        style_music = choice(self.style2music[style_names])
        same_style_music = choice(self.style2music[style_names])
        other_style_music = choice(self.style2music[other_style])

        style_music_features = self.music_features[style_music]
        same_style_music_features = self.music_features[same_style_music]
        other_style_music_features = self.music_features[other_style_music]

        begin_index = np.random.randint(0, style_music_features.shape[0] - self.arg.input_n + 1)
        style_music_windows = torch.tensor(style_music_features[begin_index:begin_index + self.arg.input_n]).float()

        begin_index = np.random.randint(0, same_style_music_features.shape[0] - self.arg.input_n + 1)
        same_style_music_windows = torch.tensor(
            same_style_music_features[begin_index:begin_index + self.arg.input_n]).float()

        begin_index = np.random.randint(0, other_style_music_features.shape[0] - self.arg.input_n + 1)
        other_style_music_windows = torch.tensor(
            other_style_music_features[begin_index:begin_index + self.arg.input_n]).float()

        motion_sample_length = self.arg.input_n + self.arg.output_n

        sequence_length = pose_features.shape[0] - motion_sample_length
        begin_index = np.random.randint(0, sequence_length + 1)

        ## CONSTRUCT TENSOR OF INPUT FEATURES ##
        pose_windows = pose_features[begin_index:begin_index + motion_sample_length]
        beat_windows = copy.deepcopy(pose_windows[:, -1])
        pose_windows = pose_windows[:, :-1]
        dist = 0
        for i in range(begin_index + motion_sample_length - 1, pose_features.shape[0]):
            if pose_features[i][-1] > 0:
                break
            dist += 1
            if dist >= motion_sample_length - 1:
                break

        for i in range(motion_sample_length - 1, -1, -1):
            if beat_windows[i] > 0:
                dist = 0
            beat_windows[i] = dist
            dist += 1

        style_window = []
        if begin_index >= 40:
            style_window += range(0, begin_index-40+1)
        if begin_index+motion_sample_length+40 <= pose_features.shape[0]:
            style_window += range(begin_index+motion_sample_length, pose_features.shape[0]-40+1)

        style_index = random.sample(style_window, 1)[0]
        context_index = random.sample(style_window, 1)[0]

        style_windows = pose_features[style_index:style_index + 40, :-1]
        style_windows = torch.tensor(style_windows).float()

        context_windows = pose_features[context_index:context_index + 40, :-1]
        context_windows = torch.tensor(context_windows).float()

        pose_windows = torch.tensor(pose_windows).float()

        beat_windows = torch.tensor(beat_windows).long()

        return pose_windows, beat_windows, style_windows, context_windows, \
               style_music_windows, same_style_music_windows, other_style_music_windows


    def __len__(self):
        if self.split == "train":
            return len(self.base_filenames) * self.arg.dataset_repeat
        else:
            return len(self.base_filenames) * 4


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Pose3dMusicDataset.modify_commandline_options(parser)
    dataset = Pose3dMusicDataset(parser.parse_args())
    a = dataset[1]
    print(a)