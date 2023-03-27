from data_preprocess.utils_ import *

from numpy import linalg as LA


def calculate_motion_beats(rotations_euler: np.ndarray, align: int = 0) -> np.ndarray:
    """
    
    The code was implemented based on the following papers:
      [1] Chieh Ho, Wei-Tze Tsai, Keng-Sheng Lin, and Homer H Chen, National Taiwan University,
          Extraction and Alignment Evaluation of Motion Beats for Street Dance, ICASSP 2013
      [2] Wei-Ta Chu, Member, IEEE, and Shang-Yin Tsai, National Chung Cheng University,
          Rhythm of Motion Extraction and Rhythm-Based Cross-Media Alignment for Dance Videos,
          IEEE Transactions on Multimedia ( Volume: 14, Issue: 1, Feb. 2012 )
    [1]'s algorithm was modified for rotations.
    
    @param rotations_euler: motions that were represented by euler angles, shape: [time, 3(euler angles)].
    @param align: matching step.
    @return: motion beats time index numpy array.
    """

    num_frames, num_channels = rotations_euler.shape
    num_joints = int(num_channels / 3)

    # Calculate the speed of each frame and the change angle between the speed vectors.
    speed = np.zeros((num_frames, num_channels))
    angle = np.zeros((num_frames, num_joints))
    speed_norm = np.zeros((num_frames, num_joints))
    for i in range(1, num_frames):
        speed[i] = rotations_euler[i] - rotations_euler[i - 1]
        for j in range(num_joints):
            v1 = speed[i - 1, j * 3:j * 3 + 3]
            v2 = speed[i, j * 3:j * 3 + 3]
            if not (np.sum(v1) == 0 and np.sum(v2) == 0):
                angle[i, j] = angle_between(v1, v2)
            speed_norm[i, j] = LA.norm(v2)

    _beats = []
    active = []
    jnts_beat = []

    # Calculate a candidate beat by matching angle's peak and a local minimum of the speed of each joint.
    for j in range(num_joints):
        if np.std(angle[:, j]) > 0:
            angle[:, j] /= np.amax(np.abs(angle[:, j]))
            speed_norm[:, j] /= np.amax(np.abs(speed_norm[:, j]))
            peak_angle = peak_detect(angle[:, j])
            zero_vel = closezero_detect(speed_norm[:, j])
            joint_beat = []
            init_frame = 0
            for zero in zero_vel:
                for idx in range(init_frame, init_frame + 20):
                    if (zero >= peak_angle[idx] - align) or (zero <= peak_angle[idx] + align):
                        joint_beat += [zero]
                        init_frame = idx
                        break
            _beats += [joint_beat]
            jnts_beat += joint_beat
            active += [j]
    vel_drop = np.zeros((num_frames, len(active)))
    jnts_beat = np.array(jnts_beat)

    # Calculate the speed drop on each beat.
    for j in range(len(active)):
        peak_vel = peak_detect(speed_norm[:, active[j]])
        for vdp in _beats[j]:
            vpk = np.where(peak_vel < vdp)[0]
            if len(vpk) > 0:
                vpk = peak_vel[vpk[-1]]
                vel_drop[vdp, j] = speed_norm[vpk, active[j]] - speed_norm[vdp, active[j]]
    vel_drop = np.sum(vel_drop, axis=1)

    # Process Velocity drops.
    min_drop = np.where(vel_drop < np.std(vel_drop))[0]
    vel_drop[min_drop] = -0.1
    drop_cross = np.where(np.diff(np.signbit(vel_drop)))[0]
    for i in range(0, len(drop_cross), 2):
        xi = drop_cross[i] - align
        xj = drop_cross[i + 1] + align + 1
        segment = vel_drop[xi:xj]
        maxs = np.where(segment > 0)[0]
        if len(maxs) > 1:
            max_id = np.where(segment == np.amax(segment))[0][0]
            for j in range(xi, xj):
                if not(j - xi == max_id) and j < vel_drop.shape[0]:
                    vel_drop[j] = -0.1
    drops = np.asarray(np.where(vel_drop > 0)[0])

    # Match the speed drop with the candidate beats in frames.
    candidate_beat = []
    for dp in drops:
        candidate = np.where(jnts_beat == dp)[0]
        if len(candidate) > 0:
            candidate_beat += [dp]
    candidate_beat = np.sort(np.unique(np.asarray(candidate_beat)))
    return candidate_beat


# region Test code.

if __name__ == '__main__':
    input = np.random.rand(1000, 3)

    output = calculate_motion_beats(input)

    print(output)

# endregion