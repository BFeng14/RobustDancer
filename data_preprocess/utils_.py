import numpy as np


def unit_vector(vector):
    """Returns the unit vector of the vector.  """

    div = np.linalg.norm(vector)
    if div != 0:
        return vector / div
    else:
        return vector


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def peak_detect(signal):
    # signal in 1D
    gradient = np.gradient(signal)
    zero_cross = np.where(np.diff(np.signbit(gradient)))[0]
    peak = []
    for i in range(0, len(zero_cross) - 2):
        xss1, _, xss2 = zero_cross[i:i + 3]
        portion = signal[xss1:xss2]
        amax = np.amax(np.abs(portion))
        idx = np.where(np.abs(portion) == amax)[0]
        peak += [(xss1 + x) for x in idx]
    peak = np.sort(np.unique(np.asarray(peak)))
    return peak


def closezero_detect(signal):
    # signal in 1D
    gradient = np.gradient(signal)
    zero_cross = np.where(np.diff(np.signbit(gradient)))[0]
    closzero = []
    for i in range(len(zero_cross) - 2):
        xss1, _, xss2 = zero_cross[i:i + 3]
        portion = signal[xss1:xss2]
        amin = np.amin(np.abs(portion))
        idx = np.where(np.abs(portion) == amin)[0]
        closzero += [(xss1 + x) for x in idx]
    return np.asarray(closzero)