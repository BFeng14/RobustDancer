import torch
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg

# See https://github.com/google/aistplusplus_api/ for installation
from aist_plusplus.features.kinetic import extract_kinetic_features
from aist_plusplus.features.manual import extract_manual_features


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


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    Code apapted from https://github.com/mseitzer/pytorch-fid

    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    mu and sigma are calculated through:
    ```
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    ```
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]  # (seq_len, 24, 3)
    return keypoints3d


def extract_feature(keypoints3d, mode="kinetic"):
    if mode == "kinetic":
        feature = extract_kinetic_features(keypoints3d)
    elif mode == "manual":
        feature = extract_manual_features(keypoints3d)
    else:
        raise ValueError("%s is not support!" % mode)
    return feature  # (f_dim,)


def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    frechet_dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    avg_dist = calculate_avg_distance(feature_list2)
    return frechet_dist, avg_dist


stylename2idx = {
            "BR": 0,
            "PO": 1,
            "LO": 2,
            "MH": 3,
            "LH": 4,
            "HO": 5,
            "WA": 6,
            "KR": 7,
            "JS": 8,
            "JB": 9,
        }

import sklearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
if __name__ == "__main__":
    import os

    # get cached motion features for the real data
    extracted_motion_path = "path_to_feature/extracted_motion_feature_20fps/"
    anno_dir = "path_to_anno_dir/aist_plusplus_final"

    seq_names = []
    seq_names += np.loadtxt(
        os.path.join(anno_dir, "splits/crossmodal_train.txt"), dtype=str
    ).tolist()
    ignore_list = np.loadtxt(
        os.path.join(anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()
    seq_names = [name for name in seq_names if name not in ignore_list]

    seq_names_val = []
    seq_names_val += np.loadtxt(
        os.path.join(anno_dir, "splits/crossmodal_val.txt"), dtype=str

    ).tolist()
    seq_names_val += np.loadtxt(
        os.path.join(anno_dir, "splits/crossmodal_test.txt"), dtype=str
    ).tolist()

    real_features = []
    labels = []
    for f in seq_names:
        kinetic = np.load(os.path.join(extracted_motion_path, f + "_kinetic.npy"))
        manual = np.load(os.path.join(extracted_motion_path, f + "_manual.npy"))
        real_features.append(np.concatenate([kinetic, manual]))
        labels.append(stylename2idx[f[1:3]])

    val_features = []
    val_labels = []
    for f in seq_names_val:
        kinetic = np.load(os.path.join(extracted_motion_path, f + "_kinetic.npy"))
        manual = np.load(os.path.join(extracted_motion_path, f + "_manual.npy"))
        val_features.append(np.concatenate([kinetic, manual]))
        val_labels.append(stylename2idx[f[1:3]])

    classifier = SVC(kernel="linear")
    classifier.fit(real_features, labels)
    # pickle.dump(classifier, open("svm_classifier.pickle", "wb"))

    pred = classifier.predict(val_features)
    print(classification_report(val_labels, pred))
