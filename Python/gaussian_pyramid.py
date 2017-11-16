import numpy as np
import cv2


# def impyramid(prevLevel, direction):
#     a = 0.375
#     k = np.array([(0.25-(a/2)), 0.25, a, 0.25, (0.25-(a/2))])[np.newaxis]
#     w = np.dot(k.T, k)
#
#     nextLevel = np.zeros(tuple(np.ceil([prevLevel/2 for x in w.shape]).astype(np.uint8)))


def gaussian_pyramid(image, N):
    gaussianPyramid = [image]

    for i in range(N-1):
        gaussianPyramid.append(cv2.pyrDown(gaussianPyramid[i]))

    return gaussianPyramid