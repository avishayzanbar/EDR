import numpy as np

from simple_pyramid_blending import simple_pyramid_blending


def verify_images(images_list):
    verified_images = []
    for image in images_list:
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        verified_images.append(image)

    return verified_images


def second_stage_improve(A, B, D):
    th = 0.8
    [A, B] = verify_images([A, B])
    grayA = ((np.sum(A ** 3, axis=2)) ** (1 / 3)) / (3 ** (1 / 3))
    grayB = ((np.sum(B ** 3, axis=2)) ** (1 / 3)) / (3 ** (1 / 3))

    mask = np.zeros(grayA.shape)
    mask[grayB > th] = grayB[grayB > th] - grayA[grayB > th]
    mask = mask > 0

    blended_image = simple_pyramid_blending(B, A, mask, D - 2)
    normalized_image = blended_image / np.max(blended_image)
    temp_map = blended_image - 1
    temp_map[temp_map < 0] = 0

    improved_mask = np.zeros(blended_image[:, :, 0].shape)
    improved_mask[(temp_map[:, :, 0] > 0) | (temp_map[:, :, 1] > 0) | (temp_map[:, :, 2] > 0)] = 1.0
    output = simple_pyramid_blending(blended_image, normalized_image, improved_mask, D - 1)

    return output