import numpy as np
import scipy.ndimage as spn
import cv2


def expand(prev_level):
    a = 0.375
    k = np.array([(0.25-(a/2)), 0.25, a, 0.25, (0.25-(a/2))])[np.newaxis]
    w = np.repeat(np.dot(k.T, k)[:, :, np.newaxis], prev_level.shape[-1], axis=2)

    next_level = np.zeros((int(np.ceil(2 * prev_level.shape[0])),
                          int(np.ceil(2 * prev_level.shape[1])),
                          prev_level.shape[2]))
    next_level[::2, ::2, :] = prev_level
    next_level = spn.correlate(next_level, w, mode='constant')

    return 4 * next_level


def laplacian_pyramid(gauss_pyr, N):
    laplace_pyr = []

    for i in range(N-1):
        laplace_pyr.append(gauss_pyr[i] - cv2.pyrUp(gauss_pyr[i+1]))
        # lapPyr.append(gaussPyr[i] - expand(gaussPyr[i+1]))

    laplace_pyr.append(gauss_pyr[-1])

    return laplace_pyr