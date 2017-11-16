import numpy as np

from reflect_image import reflect_image
from gaussian_pyramid import gaussian_pyramid
from laplacian_pyramid import laplacian_pyramid
from pyramid_blend import expand_level


def simple_pyramid_blending(A, B, R, N):
    [rows, cols, _] = A.shape

    reflected_A = reflect_image(A, rows, cols)
    reflected_B = reflect_image(B, rows, cols)
    reflected_R = reflect_image(R, rows, cols)

    gauss_pyramid_A = gaussian_pyramid(reflected_A, N)
    gauss_pyramid_B = gaussian_pyramid(reflected_B, N)
    gauss_pyramid_R = gaussian_pyramid(reflected_R, N)

    laplace_pyramid_A = laplacian_pyramid(gauss_pyramid_A, N)
    laplace_pyramid_B = laplacian_pyramid(gauss_pyramid_B, N)

    blended_pyramid = []
    for idx in range(N):
        blended_pyramid.append(np.multiply(laplace_pyramid_A[idx], (1 - gauss_pyramid_R[idx][..., None])) +
                               np.multiply(laplace_pyramid_B[idx], (gauss_pyramid_R[idx][..., None])))

    blended_image = blended_pyramid[-1]
    for i in range(N - 2, -1, -1):
        blended_image = expand_level(blended_image, blended_pyramid[i])

    return blended_image[:rows, :cols, :]
