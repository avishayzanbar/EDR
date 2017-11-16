from scipy.signal import medfilt2d
from rgb2ycrcb import *

factors = [2, 1.2, 1, 1.5, 2.3, 2.9, 3.5, 5, 8]


def brighten_img(image, factor):
    if factor == 1.5:
        bright_img = np.power(image, np.exp((factor - 2.5) * image))
    else:
        bright_img = np.power(np.tanh(factor * image), 1.1)

    # if factor > 7:
    #     ycbcr = rgb2ycrcb(bright_img)
    #     ycbcr[1, :, :] = medfilt2d(ycbcr[1, :, :], 5)
    #     ycbcr[2, :, :] = medfilt2d(ycbcr[2, :, :], 5)
    #     bright_img = ycrcb2rgb(ycbcr)
    return bright_img


def darken_img(image, factor):
    return np.power(image, factor)


def create_images_set(image):
    new_images, bright_action = [], False

    for factor in factors:
        if bright_action:
            new_img = brighten_img(image=image, factor=factor)
        else:
            if factor == 1:
                new_img = image
                bright_action = True
            else:
                new_img = darken_img(image, factor)

        new_images.append(new_img)

    return np.array(new_images)

