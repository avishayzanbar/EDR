import numpy as np
import scipy.ndimage as spn
import scipy.signal as sps
import cv2
from dct_map import dct_map


def display_img(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    new_size = tuple([int(x/3) for x in list(image.shape[:2])])
    img = cv2.resize(image, new_size[::-1])
    cv2.imshow(name, img)


def get_image_gradients(image, hfilter):
    alpha, beta, eps = 0.1, 0.8, 1e-10
    if len(image.shape) == 3 and image.shape[2] == 3:
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        singleChromeImg = ((np.sum(image[:, :, :] ** 3, axis=2)) ** (1/3)) / (3 ** (1/3))
    else:
        singleChromeImg = image

    filteredImg = spn.correlate(singleChromeImg, hfilter, mode='constant')
    filteredImg = sps.medfilt2d(filteredImg, (13, 13))

    weightsFunc = np.zeros(filteredImg.shape)
    weightsFunc[filteredImg > 0.5] = 1 - filteredImg[filteredImg > 0.5]
    weightsFunc[filteredImg <= 0.5] = filteredImg[filteredImg <= 0.5]

    gx, gy = np.zeros(filteredImg.shape), np.zeros(filteredImg.shape)
    gx[2:, :] = 0.5 * (filteredImg[2:, :] - filteredImg[:-2, :])
    gy[:, 2:] = 0.5 * (filteredImg[:, 2:] - filteredImg[:, :-2])

    imageGrads = np.sqrt(gx ** 2 + gy **2)
    imageGrads[imageGrads == 0] = eps

    phi = (alpha / imageGrads) * ((imageGrads / alpha) ** beta)

    return imageGrads * phi * weightsFunc





