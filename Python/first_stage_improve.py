import numpy as np

from simple_pyramid_blending import simple_pyramid_blending


def first_stage_improve(map, th, b_img, b_img_s, blend_depth):
    first_stage_map = np.zeros(map.shape)
    first_stage_map[map > th * 0.9] = 1

    output = simple_pyramid_blending(b_img, b_img_s, first_stage_map, blend_depth)
    return output