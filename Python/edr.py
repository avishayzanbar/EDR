import numpy as np
import scipy.ndimage as spn

from reflect_image import reflect_image
from calc_gradients_image import get_image_gradients
from gaussian_pyramid import gaussian_pyramid
from laplacian_pyramid import laplacian_pyramid
from pyramid_blend import pyramid_build


def new_img_size(img):
    m, n, _ = img.shape
    return 2 ** np.ceil(np.log2(m)).astype(np.uint8), 2 ** np.ceil(np.log2(n)).astype(np.uint8)


def fgaussian(size, sigma=0.5):
    m, n = size
    h, k = m/2, n/2
    x, y = np.mgrid[np.ceil(-h):np.ceil(h), np.ceil(-k):np.ceil(k)]

    h = np.exp(-(x**2 + y**2)/(2*sigma**2))
    return h / np.sum(h)


def edr(images_set, blend_depth):
    # Init
    blend_depth_s = blend_depth - 3
    filter_size = 27

    hfilter = fgaussian((filter_size, filter_size))

    # Calculate Gradiants, Gaussian Pyramid, Laplacian Pyramid for each image
    first_img, images_gradients, lap_pyramid, lap_pyramid_s = True, [], [], []
    rows, cols, new_rows, new_cols = None, None, None, None
    for idx, image in enumerate(images_set):
        if first_img:
            [rows, cols, _] = image.shape
            new_rows, new_cols = new_img_size(image)
            first_img = False

        reflected_image = reflect_image(image, rows, cols, new_rows, new_cols)
        images_gradients.append(get_image_gradients(image, hfilter))
        gauss_pyramid = gaussian_pyramid(reflected_image, blend_depth)
        lap_pyramid.append(laplacian_pyramid(gauss_pyramid, blend_depth))
        lap_pyramid_s.append(lap_pyramid[idx][:blend_depth_s])
        lap_pyramid_s[idx][blend_depth_s - 1] = gauss_pyramid[blend_depth_s - 1]

        del gauss_pyramid

    # Get Maps
    images_gradients = np.array(images_gradients)
    values_map, locations_map = np.max(images_gradients, axis=0), np.argmax(images_gradients, axis=0)
    del images_gradients

    locations_map = spn.correlate(locations_map.astype(np.float32), hfilter, mode='constant')
    reflected_map = reflect_image(locations_map, rows, cols, new_rows, new_cols)  # .astype(np.float32)

    # Blend pyramid into 2 representing images
    blended_img, blended_img_s = pyramid_build(lap_pyramid, lap_pyramid_s, reflected_map, blend_depth, blend_depth_s)
    blended_img = blended_img[:rows, :cols, :]
    blended_img_s = blended_img_s[:rows, :cols, :]

    return blended_img, blended_img_s, values_map


