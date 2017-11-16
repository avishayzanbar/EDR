import numpy as np
import cv2, time


def display_img(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    new_size = tuple([int(x/3) for x in list(image.shape[:2])])
    img = cv2.resize(image, new_size[::-1])
    cv2.imshow(name, img)


def dct_map(image):
    zero_factor, eps = 0.1, 0.1
    gx, gy = np.zeros(image.shape), np.zeros(image.shape)
    gx[2:, :] = 0.5 * (image[2:, :] - image[:-2, :])
    gy[:, 2:] = 0.5 * (image[:, 2:] - image[:, :-2])

    imageGrads = np.sqrt(gx ** 2 + gy ** 2)
    imageGrads[imageGrads == 0] = eps
    map_dct = cv2.dct(imageGrads)
    zero_point = int(zero_factor * len(map_dct.ravel()))

    m, _ = np.indices((len(map_dct.ravel()), 1))
    mm = m.reshape(map_dct.shape)

    # ts = time.time()
    hh = np.concatenate([np.diagonal(mm[::-1, :], k)[::(2 * (k % 2) - 1)] for k in range(1 - mm.shape[0], mm.shape[0])])
    # te = time.time()
    # print(hh)
    # print(hh.shape)
    # print(te-ts)

    # print(map_dct)
    map_dct_vect = map_dct.ravel()
    map_dct_vect[-hh[:zero_point]] = 0.0
    map_dct_res = map_dct_vect.reshape(map_dct.shape)
    # print(map_dct_res)
    # exit()

    ret_map = cv2.dct(map_dct_res, cv2.DCT_INVERSE)
    # display_img(ret_map, "inv map")
    return ret_map
