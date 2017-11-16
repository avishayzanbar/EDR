import numpy as np
import cv2


def rgb2ycrcb(im_rgb):  # TODO: set the function the same as in Matlab
    # origT = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]])
    # origOffset = np.array([[16], [128], [128]])
    # scaleFactorT = 1 / 255.0
    # scaleFactorOffset = 1 / 255.0
    #
    # T = scaleFactorT * origT
    # offset = scaleFactorOffset * origOffset
    #
    # ycbcr = np.zeros(img.shape)
    # im_rgb = img # im_rgb.astype(np.float32)
    # im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    # im_ycrcb = cv2.cvtColor((255.0 * im_rgb).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    # im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32) / 255.0
    # im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    # im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]

    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycrcb[:, :, 0] = (im_ycrcb[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycrcb[:, :, 1:] = (im_ycrcb[:, :, 1:] * (240 - 16) + 16) / 255.0
    return im_ycrcb


def ycrcb2rgb(im_ycrcb):
    im_ycrcb[:, :, 0] = (im_ycrcb[:, :, 0] * 255.0 - 16) / (235 - 16)  # to [0, 1]
    im_ycrcb[:, :, 1:] = (im_ycrcb[:, :, 1:] * 255.0 - 16) / (240 - 16)  # to [0, 1]
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCrCb2RGB)
    return im_rgb