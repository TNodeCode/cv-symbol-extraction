import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import random
import torch


def overlap(boxes1, boxes2):
    # top left corners of all combinations
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

    # bottom right corners of all combinations
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # width and hight of overlap area of boxes1 and boxes2.
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]#
    return torch.all(inter == 0)


def provide_random_coordinates(background, img):
    """ Provides random coordinates for placement of image.

    :param background: Background image.
    :param img: Image to be placed.
    :return: Random x and y coordinates.
    """
    h_img, w_img, _ = img.shape
    h_background, w_background, _ = background.shape
    y_start = random.randint(5, h_background-h_img-5)
    x_start = random.randint(5, w_background-w_img-5)
    return [x_start, y_start, x_start+w_img, y_start+h_img]


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
    x = bbox[1] + x_start - 5
    y = bbox[0] + y_start - 5 
    width = bbox[3] - bbox[1] + 5
    height = bbox[2] - bbox[0] + 5

    # draw bbox on the image
    # plt.gca().add_patch(Rectangle((x, y), width, height,
    #                               linewidth=1, edgecolor='r', facecolor='none'))
    return x, y, x + width, y + height


def create_label_object(x, y, width, height, label):
    """ Creates label for YOLO algorithm

    :param file:
    :param x:
    :param y:
    :param width:
    :param height:
    :return:
    """
    label_txt = ''
    label_txt += str(x) + ','  # x1
    label_txt += str(y) + ','  # y1
    label_txt += str(width) + ','  # x2
    label_txt += str(height) + ','  # y2
    label_txt += str(label) + ' '
    return label_txt


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
    img[img < 128] = 0
    img[img >= 128] = 255
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
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

def random_color(image):
    h, w, _ = image.shape
    color = np.zeros((h, w, 3))
    color[:, : , 0] += random.randint(0, 255)
    color[:, : , 1] += random.randint(0, 255)
    color[:, : , 2] += random.randint(0, 255)
    image[:, :, :3] = color
    return image

def place_image_on_background(label, image, background, coordinates, i):
    # Prepare and resize background
    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    background = resize(background, 450)

    # Prepare image.
    image = swap_black_white(image)

    # resize and find coordinates
    img = resize(image, random.randint(
        int(image.shape[0]/2), int(image.shape[0]*1.3)))
    co = torch.tensor(provide_random_coordinates(
        background, img)).reshape((1, 4))
    while not overlap(co, coordinates):
        img = resize(image, random.randint(
            int(image.shape[0]/2), int(image.shape[0]*1.3)))
        co = torch.tensor(provide_random_coordinates(
            background, img)).reshape((1, 4))
    coordinates[i] = co

    mask = get_img_mask(img)
    img[:, :, 3] = mask
    img = random_color(img)
    background_alpha = 1.0 - mask

    # get placement coordinates
    x_start, y_start, x_end, y_end = co[0].tolist()

    # use numpy indexing to place the resized image in the background image
    for c in range(0, 3):
        background[y_start:y_end, x_start:x_end, c] = \
            (mask * img[:, :, c] + background_alpha *
             background[y_start:y_end, x_start:x_end, c])

    x1, y1, x2, y2 = create_bounding_box(mask, x_start, y_start)
    label_text = create_label_object(x1, y1, x2, y2, label)
    return background, label_text
