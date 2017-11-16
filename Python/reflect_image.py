import numpy as np


def reflect_image(image, oldRows, oldCols, newRows=None, newCols=None):
    if newRows is None or newCols is None:
        newRows, newCols = new_img_size(image)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    outputImage = np.zeros((newRows, newCols, image.shape[-1]))

    outputImage[:oldRows, :oldCols, :] = image
    outputImage[oldRows:, :oldCols, :] = np.flip(image[2 * oldRows - newRows:, :, :], axis=0) # Down
    outputImage[:oldRows, oldCols:, :] = np.flip(image[:, 2 * oldCols - newCols:, :], axis=1) # Right
    outputImage[oldRows:, oldCols:, :] = np.flip(np.flip(image[2 * oldRows - newRows:, 2 * oldCols - newCols:, :],
                                                         axis=0), axis=1)

    '''
    print('1', outputImage[:oldRows, :oldCols, :].shape, image.shape)
    print('2', outputImage[oldRows:, :oldCols, :].shape, image[:newRows - oldRows, :, :].shape)
    print('3', outputImage[:oldRows, oldCols:, :].shape, image[:, :newCols - oldCols, :].shape)
    print('4', outputImage[oldRows:, oldCols:, :].shape, image[:newRows - oldRows, 2 * oldCols - newCols - 1:-1, :].shape)

    print('2', outputImage[oldRows:, :oldCols, :].shape, np.flip(image[:2 * oldRows - newRows, :, :], axis=0).shape)
    outputImage[oldRows:, :oldCols, :] = np.flip(image[:2 * oldRows - newRows, :, :], axis=0)
    print('3', outputImage[:oldRows, oldCols:, :].shape, np.flip(image[:, 2 * oldCols - newCols, :], axis=0).shape)
    outputImage[:oldRows, oldCols:, :] = np.flip(image[:, 2 * oldCols - newCols, :], axis=0)
    print('4', outputImage[oldRows:, oldCols:, :].shape, np.flip(np.flip(image[:2 * oldRows - newRows, 2 * oldCols - newCols:-1, :], axis=1), axis=0).shape)
    outputImage[oldRows:, oldCols:, :] = np.flip(np.flip(image[:2 * oldRows - newRows, 2 * oldCols - newCols:-1, :], axis=1), axis=0)  
    '''

    return np.squeeze(outputImage)


def new_img_size(img):
    m, n = img.shape[:2]
    return 2 ** np.ceil(np.log2(m)).astype(np.uint8), 2 ** np.ceil(np.log2(n)).astype(np.uint8)
