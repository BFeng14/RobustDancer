from absl import app
from absl import flags
from absl import logging

import os
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal
from aist_plusplus.loader import AISTDataset



import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--anno_dir', type=str, default='path_to_anno_dir/aist_plusplus_final',
    help='Path to the AIST++ annotation files.')
parser.add_argument(
    '--audio_dir', type=str, default='path_to_wav',
    help='Path to the AIST wav files.')
parser.add_argument(
    '--audio_cache_dir', type=str, default='path_to_feature/audio_feature_20',
    help='Path to cache dictionary for audio features.')
parser.add_argument(
    '--motion_cache_dir', type=str, default='path_to_feature/motion_feature_20',
    help='Path to cache dictionary for audio features.')
parser.add_argument(
    '--split', type=str, default='testval',
    help='Whether do training set or testval set.')
parser.add_argument(
    '--result_files', type=str, default='path_to_results',
    )

RNG = np.random.RandomState(42)
FLAGS = parser.parse_args()

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    transl = motion[:, :, :3]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 3:219], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    return keypoints3d


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=3)  # 4 for 20FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


def alignment_score(music_beats, motion_beats, sigma=1):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)


from data_preprocess.motion_feature import calculate_motion_beats
def main(_):
    import glob
    import tqdm
    from smplx import SMPL

    # set smpl
    smpl = SMPL(model_path="path_to_smpl/models", gender='MALE', batch_size=1)

    # create list
    seq_names = []
    if "train" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_train.txt"), dtype=str
        ).tolist()
    if "val" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_val.txt"), dtype=str
        ).tolist()
    if "test" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_test.txt"), dtype=str
        ).tolist()
    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()
    seq_names = [name for name in seq_names if name not in ignore_list]

    # calculate score on real data
    # dataset = AISTDataset(FLAGS.anno_dir)
    # n_samples = len(seq_names)
    # beat_scores = []
    # for i, seq_name in enumerate(seq_names):
    #     logging.info("processing %d / %d" % (i + 1, n_samples))
    #     # get real data motion beats
    #     smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
    #         dataset.motion_dir, seq_name)
    #     smpl_trans /= smpl_scaling
    #     keypoints3d = smpl.forward(
    #         global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    #         body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    #         transl=torch.from_numpy(smpl_trans).float(),
    #     ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    #     motion_beats = motion_peak_onehot(keypoints3d)
    #     # get real data music beats
    #     audio_name = seq_name.split("_")[4]
    #     audio_feature = np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy"))
    #     audio_beats = audio_feature[:keypoints3d.shape[0], -1] # last dim is the music beats
    #     # get beat alignment scores
    #     beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
    #     beat_scores.append(beat_score)
    # print ("\nBeat score on real data: %.3f\n" % (sum(beat_scores) / n_samples))

    # calculate score on generated motion data
    result_files = sorted(glob.glob(FLAGS.result_files+"/*.npy"))

    n_samples = len(result_files)
    beat_scores = []
    beat_nums = []
    for result_file in tqdm.tqdm(result_files):
        result_motion = np.load(result_file)[None, ...]  # [1, 120 + 1200, 225]
        keypoints3d = recover_motion_to_keypoints(result_motion, smpl)
        keypoints3d_root = keypoints3d[:, 0:1, :]
        keypoints3d_del_root = keypoints3d - keypoints3d_root
        keypoints3d_del_root = keypoints3d_del_root
        motion_beats = motion_peak_onehot(keypoints3d_del_root)[40:400+40]
        beat_nums.append(motion_beats.astype(np.float).sum())

        audio_name = result_file[-13:-9]
        audio_feature = np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy"))
        audio_beats = audio_feature[:, -1]  # last dim is the music beats
        # calculate_motion_beats(keypoints3d_del_root)
        beat_score = alignment_score(audio_beats[40:400+40], motion_beats, sigma=1)
        beat_scores.append(beat_score)
    print ("\nBeat score on generated data: %.3f\n" % (sum(beat_scores) / n_samples))
    print(np.average(np.array(beat_nums)))

if __name__ == '__main__':
    app.run(main)
