import numpy as np
import cv2

from gaussian_pyramid import gaussian_pyramid


def expand_level(prev_level, next_level):
    expandedLevel = cv2.pyrUp(prev_level)
    return expandedLevel + next_level


def blend_action(images, mask):
    blended_output = np.zeros(images[0].shape)  # np.multiply(np.ones(images[0].shape), images[0])
    for i, image in enumerate(images):
        if i == 0:
            m = mask < (i + 0.5)  # TODO: mask is not considered as in Matlab
            # m = np.multiply(image[mask < (i + 0.5)], mask[mask < (i + 0.5)]) # TODO: mask is not considered as in Matlab
        elif i == len(images) - 1:
            m = mask >= (i - 0.5)  # TODO: mask is not considered as in Matlab
        else:
            m = np.bitwise_and(mask >= (i - 0.5), mask < (i + 0.5))
        # blended_output[m] += np.multiply(image[m], (np.repeat(mask[m][:, np.newaxis], 3, axis=1) / (i + 1)))
        blended_output[m] = np.multiply(image[m], ((1 + mask[m][..., None]) / (i + 1)))

    return blended_output


def pyramid_build(lapPyr, lapPyrS, map, N, Ns):
    gaussPyrR = gaussian_pyramid(map, N)
    gaussPyrRs = gaussPyrR[:Ns]

    lapPyr = np.array(lapPyr)
    lapPyrS = np.array(lapPyrS)

    blendedPyramid, blendedPyramidS = [], []
    for iCell in range(len(gaussPyrR)):
        blendedPyramid.append(blend_action(lapPyr[:, iCell], gaussPyrR[iCell]))
        if iCell <= Ns - 1:
            blendedPyramidS.append(blend_action(lapPyrS[:, iCell], gaussPyrRs[iCell]))

    blendedImage, blendedImageS = blendedPyramid[-1], blendedPyramidS[-1]
    for i in range(N - 2, -1, -1):
        blendedImage = expand_level(blendedImage, blendedPyramid[i])

        if i < Ns - 1:
            blendedImageS = expand_level(blendedImageS, blendedPyramidS[i])

    return blendedImage, blendedImageS