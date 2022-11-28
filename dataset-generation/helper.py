import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import random
import glob
import os


def provide_random_coordinates(background, img):
    """ Provides random coordinates for placement of image.

    :param background: Background image.
    :param img: Image to be placed.
    :return: Random x and y coordinates.
    """
    h_img, w_img, _ = img.shape
    h_background, w_background, _ = background.shape
    y_start = random.randint(h_img, h_background-h_img)
    x_start = random.randint(w_img, w_background-w_img)
    return x_start, x_start+w_img, y_start, y_start+h_img


def create_bounding_box(mask, x_start, y_start):
    """ Creates bounding box around placed image.

    :param mask: Mask of placed image.
    :param x_start: X coordinate (bottom left) of placed image in background
    :param y_start: Y coordinate (bottom left) of placed image in background
    :return: Computed labels for YOLO-algorithm
    """

    # Compute bounding box dimensions
    nz = np.nonzero(mask)
    bbox = [np.min(nz[0]), np.min(nz[1]), np.max(nz[0]), np.max(nz[1])]

    # Compute anchor points in Background image
    x = bbox[1] + x_start
    y = bbox[0] + y_start
    width = bbox[3] - bbox[1]
    height = bbox[2] - bbox[0]

    # draw bbox on the image
    plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none'))
    return int(x + width / 2 + height / 2), int(y + width / 2 + height / 2), width, height


def create_label(file, x, y, width, height):
    """ Creates label for YOLO algorithm

    :param file:
    :param x:
    :param y:
    :param width:
    :param height:
    :return:
    """
    pass


def get_img_mask(img):
    """ Creates mask of given image. A mask will allow us to focus on the specific portion of the input image,
    in our case - a symbol.

    :param img: Input image.
    :return: Mask of image.
    """
    mask = img.copy()[:, :, 0]
    mask[mask > 0] = 1
    return mask


def swap_black_white(img):
    """ Our input images are black on white background. For mimicking standard chalk,
    we swap to have white on black background.

    :param img: Input image.
    :return: Image with swapper color channels.
    """
    img[img <= 5] = 240
    img[img > 240] = 0
    img[img == 240] = 255
    return img


def resize(img, wanted_width):
    """ Resizes background while keeping dimensions

    :param img: Input image.
    :param wanted_width: Desired size of horizontal axis.
    :return: Scaled Image.
    """
    scale = wanted_width / img.shape[0]
    output_height = int(img.shape[1] * scale)
    img = cv2.resize(img, (output_height, wanted_width))
    return img


def place_image_on_background(label, img, background):
    print(label)
    # Prepare background and image.
    img = swap_black_white(img)
    # resize image
    img = resize(img, random.randint(int(img.shape[0]/2), int(img.shape[0]*2)))
    mask = get_img_mask(img)
    background = resize(background, 450)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[:, :, 3] = mask
    background_alpha = 1.0 - mask

    # get placement coordinates
    x_start, x_end, y_start, y_end = provide_random_coordinates(background, img)

    # use numpy indexing to place the resized image in the background image
    for c in range(0, 3):
        background[y_start:y_end, x_start:x_end, c] = \
            (mask * img[:, :, c] + background_alpha * background[y_start:y_end, x_start:x_end, c])

    x, y, width, height = create_bounding_box(mask, x_start, y_start)
    print(f"Fake labels for YOLO - to be implemented and stored in file Label: {label}, X_center:{x}, Y_center:{y}, "
          f"Width: {width}, Height: {height}")
    return background


def place_images_on_background(background):
    image_tuples = [(cv2.imread(file), os.path.basename(file)[0]) for file in glob.glob("images/*.jpg")]
    for image, label in image_tuples:
        background = place_image_on_background(label, image, background)
    return background
