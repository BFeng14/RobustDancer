from tqdm import tqdm
import os
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import librosa
import torch
from aist_plusplus.loader import AISTDataset
import multiprocessing
import functools
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
    '--motion_cache_dir', type=str, default='path_to_feature/motion_feature_20_beat_3',
    help='Path to cache dictionary for audio features.')
parser.add_argument(
    '--split', type=str, default='traintestval',
    help='Whether do training set or testval set.')
RNG = np.random.RandomState(42)

FLAGS = parser.parse_args()


def close_tfrecord_writers(writers):
    for w in writers:
        w.close()


def write_tfexample(writers, tf_example):
    random_writer_idx = RNG.randint(0, len(writers))
    writers[random_writer_idx].write(tf_example.SerializeToString())


def load_cached_audio_features(seq_name):
    audio_name = seq_name.split("_")[-2]
    return np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")), audio_name


def cache_audio_features(seq_names):
    FPS = 20
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(audio_name):
        """Get tempo (BPM) for a music by parsing music name."""
        assert len(audio_name) == 4
        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[0:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else:
            assert False, audio_name

    audio_names = list(set([seq_name.split("_")[-2] for seq_name in seq_names]))

    for audio_name in tqdm(audio_names):
        save_path = os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")
        if os.path.exists(save_path):
            continue
        data, _ = librosa.load(os.path.join(FLAGS.audio_dir, f"{audio_name}.wav"), sr=SR)
        envelope = librosa.onset.onset_strength(data, sr=SR)  # (seq_len,)
        mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
        chroma = librosa.feature.chroma_cens(
            data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
            start_bpm=_get_tempo(audio_name), tightness=100)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)

        audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)
        np.save(save_path, audio_feature)


def recover_motion_to_keypoints(smpl_poses, smpl_trans, smpl_model):
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]  # (seq_len, 24, 3)
    return keypoints3d


def extract_foot_contact(keypoints3d, velfactor=1.):
    # [
    #     "nose",
    #     "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    #     "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    #     "left_knee", "right_knee", "left_ankle", "right_ankle"
    # ]
    l_foot = keypoints3d[:, -2, :]
    r_foot = keypoints3d[:, -1, :]
    lfoot_xyz = (l_foot[1:, :] - l_foot[:-1, :]) ** 2
    lfoot = np.sqrt(np.sum(lfoot_xyz, axis=-1))
    lfoot = scipy.signal.savgol_filter(lfoot, 9, 2)
    contacts_l = (lfoot < velfactor)

    rfoot_xyz = (r_foot[1:, :] - r_foot[:-1, :]) ** 2
    rfoot = np.sqrt(np.sum(rfoot_xyz, axis=-1))
    rfoot = scipy.signal.savgol_filter(rfoot, 9, 2)
    contacts_r = (rfoot < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.expand_dims(np.concatenate([contacts_l, contacts_l[-1:]], axis=0), axis=-1)
    contacts_r = np.expand_dims(np.concatenate([contacts_r, contacts_r[-1:]], axis=0), axis=-1)

    return contacts_l, contacts_r


from data_preprocess.motion_feature import calculate_motion_beats
def detect_beat(smpl_poses, seq_len):
    smpl_poses_euler = smpl_poses.as_euler('XYZ').reshape(seq_len, 24, 3).reshape(seq_len, -1)
    beats = calculate_motion_beats(smpl_poses_euler, 3)
    beats_onehot = np.zeros(seq_len)
    beats_onehot[beats.tolist()] = 1
    return np.expand_dims(beats_onehot, axis=-1)


def mirror_aug(smpl_trans, smpl_poses, seq_len, axis="X"):
    joint_names = [
        "root",
        "l_hip", "r_hip", "belly",
        "l_knee", "r_knee", "spine",
        "l_ankle", "r_ankle", "chest",
        "l_toes", "r_toes", "neck",
        "l_inshoulder", "r_inshoulder",
        "head", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow",
        "l_wrist", "r_wrist",
        "l_hand", "r_hand",
    ]

    smpl_poses_euler = smpl_poses.as_euler('XYZ').reshape(seq_len, 24, 3)

    if axis == "X":
        signs = np.array([1, -1, -1])
    if axis == "Y":
        signs = np.array([-1, 1, -1])
    if axis == "Z":
        signs = np.array([-1, -1, 1])

    new_smpl_trans = smpl_trans * (-signs)
    l_joints_id = [i for i, joint in enumerate(joint_names) if joint.split("_")[0] == "l"]
    r_joints_id = []
    for idx in l_joints_id:
        name = joint_names[idx]
        r_name = "r_"+name.split("_")[1]
        r_joints_id.append(joint_names.index(r_name))

    new_smpl_poses = np.copy(smpl_poses_euler)
    new_smpl_poses[:, l_joints_id, :] = smpl_poses_euler[:, r_joints_id, :]
    new_smpl_poses[:, r_joints_id, :] = smpl_poses_euler[:, l_joints_id, :]
    new_smpl_poses = new_smpl_poses * signs

    new_smpl_poses = R.from_euler('XYZ', new_smpl_poses.reshape(-1, 3))
    return new_smpl_trans, new_smpl_poses


def write_data(smpl_trans, smpl_poses, extracts, seq_name, seq_len):
    smpl_poses = smpl_poses.as_matrix().reshape(seq_len, -1)
    smpl_motion = np.concatenate([smpl_trans, smpl_poses, extracts], axis=-1)
    sample_index = [i * 3 for i in range(smpl_motion.shape[0] // 3)]
    smpl_motion_downsample = smpl_motion[sample_index]
    np.save(os.path.join(FLAGS.motion_cache_dir, f"{seq_name}.npy"), smpl_motion_downsample)


def process_fun(seq_name, motion_dir, keypoint3d_dir):
    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
        motion_dir, seq_name)
    smpl_trans /= smpl_scaling
    seq_len = smpl_poses.shape[0]
    keypoint3d = AISTDataset.load_keypoint3d(keypoint3d_dir, seq_name, use_optim=False)
    contacts_l, contacts_r = extract_foot_contact(keypoint3d)
    smpl_poses = R.from_rotvec(
        smpl_poses.reshape(-1, 3))
    beats = detect_beat(smpl_poses, seq_len)
    extracts = np.concatenate([contacts_l, contacts_r, beats], axis=-1)

    write_data(smpl_trans, smpl_poses, extracts, seq_name, seq_len)

    smpl_trans_mirror, smpl_poses_mirror = mirror_aug(smpl_trans, smpl_poses, seq_len)
    extracts_mirror = np.concatenate([contacts_r, contacts_l, beats], axis=-1)
    write_data(smpl_trans_mirror, smpl_poses_mirror, extracts_mirror, seq_name + "mirror", seq_len)


def main():
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

    # create audio features
    # print("Pre-compute audio features ...")
    # os.makedirs(FLAGS.audio_cache_dir, exist_ok=True)
    # cache_audio_features(seq_names)

    # exit(0)
    os.makedirs(FLAGS.motion_cache_dir, exist_ok=True)

    # load data
    dataset = AISTDataset(FLAGS.anno_dir)

    process = functools.partial(process_fun, motion_dir=dataset.motion_dir, keypoint3d_dir=dataset.keypoint3d_dir)
    pool = multiprocessing.Pool(8)
    pool.map(process, seq_names)


if __name__ == '__main__':
    # app.run(main)
    main()