import os
import time

from create_images_set import create_images_set
from rgb2ycrcb import *
from edr import edr
from first_stage_improve import first_stage_improve
from second_stage_improve import second_stage_improve


def display_img(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    new_size = tuple([int(x/3) for x in list(image.shape[:2])])
    img = cv2.resize(image, new_size[::-1])
    cv2.imshow(name, img)


def load_input_img(name):
    if not os.path.isfile(name):
        print("Failed to locate", name)
        exit(1)

    image = np.float32(cv2.imread(name, -1)) / 255.0
    if not image.size:
        print("Failed to open img", name)
        exit(1)

    return image


print('Starting EDR...')
save_output = True
file_path = "images/UNADJUSTEDNONRAW_thumb_8470.jpg"
# file_path = "/Users/avishayzanbar/Dropbox/EDR/MatlUNADJUSTEDNONRAW_thumb_9092ab/code/images/set1/PA032747.JPG"
input_img = load_input_img(file_path)

ts = time.time()
images_set = create_images_set(input_img)
te = time.time()
print('New set of images created in %.2f seconds' % (te-ts))

blending_depth, threshold = 9, 2 * 0.0015
ts = time.time()
blended_image, blended_image_s, values_map = edr(images_set, blending_depth)
te = time.time()
print('Main EDR stage done in %.2f seconds' % (te - ts))

ts = time.time()
improved_blended_img = first_stage_improve(values_map, threshold, blended_image, blended_image_s, blending_depth)
te = time.time()
print('First stage improvement done in %.2f seconds' % (te - ts))

ts = time.time()
edr_image = second_stage_improve(input_img, improved_blended_img, blending_depth)
te = time.time()
print('Final stage improvement done in %.2f seconds' % (te - ts))
# display_img(edr_image, "EDR")
# cv2.waitKey(0)

if save_output:
    save_img = edr_image
    save_img[save_img < 0.] = 0.
    save_img[save_img > 1.] = 1.
    cv2.imwrite(file_path.strip(".JPG") + "_EDR_py.jpg", (255 * edr_image).astype(np.uint8))
    print('EDR Image saved')

print('Done.')
cv2.destroyAllWindows()