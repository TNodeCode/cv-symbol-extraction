import numpy as np
import cv2
import random
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def preprossesing_background(background):
    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    background, _ = resize(background, 700)
    return background


def preprossesing_image(image, labels):

    image = swap_black_white(image)
    mask = get_img_mask(image)
    nz = np.nonzero(mask)
    min_y, max_y = [np.min(nz[0]), np.max(nz[0])+1]
    # min_x, max_x = [np.min(nz[1]), np.max(nz[1])]

    image = image[min_y:max_y]  # min_x:max_x if bounding

    # For resizeing ??
    scale = 1
    # scale = ((1-((max_y - min_y) / 300)) * 1.5) **2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    bboxes = []
    for bbox in labels:
        x1 = int(bbox[0]) * scale
        y1 = (int(bbox[1]) - min_y) * scale
        x2 = int(bbox[2]) * scale
        y2 = (int(bbox[3]) - min_y) * scale
        label = bbox[4]
        bboxes.append([x1, y1, x2, y2, label])

    return image, bboxes


def overlap(boxes1, boxes2):
    # top left corners of all combinations
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

    # bottom right corners of all combinations
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # width and hight of overlap area of boxes1 and boxes2.
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]#
    return torch.all(inter == 0)


def provide_random_coordinates(background, image, coordinates, i):
    """ Provides random coordinates for placement of image.

    :param background: Background image.
    :param img: Image to be placed.
    :return: Random x and y coordinates.
    """

    h_img, w_img, _ = image.shape
    h_background, w_background, _ = background.shape

    y_start = random.randint(5, h_background-h_img-5)
    x_start = random.randint(5, w_background-w_img-5)
    co = torch.tensor([x_start, y_start, x_start+w_img,
                      y_start+h_img]).reshape((1, 4))

    flag = False
    counter = 0
    while not overlap(co, coordinates):
        y_start = random.randint(5, h_background-h_img-5)
        x_start = random.randint(5, w_background-w_img-5)
        co = torch.tensor([x_start, y_start, x_start+w_img,
                          y_start+h_img]).reshape((1, 4))
        counter += 1
        if counter >= 100:
            flag = True
            break

    coordinates[i] = co

    return [x_start, y_start, x_start+w_img, y_start+h_img, flag]


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
    x1 = bbox[1] + x_start
    y1 = bbox[0] + y_start
    x2 = bbox[3] + x_start
    y2 = bbox[2] + y_start

    # draw bbox on the image
    # plt.gca().add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
    #                               linewidth=1, edgecolor='r', facecolor='none'))
    return x1, y1, x2, y2


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


def create_label_objects(x1_start, y1_start, labels, scale):
    label_txt = ''
    for bbox in labels:
        x1 = int(bbox[0] * scale) + x1_start
        y1 = int(bbox[1] * scale) + y1_start
        x2 = int(bbox[2] * scale) + x1_start
        y2 = int(bbox[3] * scale) + y1_start
        label = bbox[4]
        # plt.gca().add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none'))

        label_txt += str(x1) + ','     # x1
        label_txt += str(y1) + ','               # y1
        label_txt += str(x2) + ','     # x2
        label_txt += str(y2) + ','               # y2
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


def resize(img, wanted):
    """ Resizes background while keeping dimensions

    :param img: Input image.
    :param wanted_width: Desired size of horizontal axis.
    :return: Scaled Image.
    """
    x, y, _ = img.shape
    if x < y:
        scale = wanted / x
        output_height = int(y * scale)
        img = cv2.resize(img, (output_height, wanted))
    else:
        scale = wanted / y
        output_width = int(x * scale)
        img = cv2.resize(img, (wanted, output_width))
    return img, scale


def random_color(image, labels):
    for label in labels:
        x1, y1, x2, y2, _ = label
        color = np.zeros((y2-y1, x2-x1, 3))
        color[:, :, 0] += random.randint(0, 255)
        color[:, :, 1] += random.randint(0, 255)
        color[:, :, 2] += random.randint(0, 255)
        image[y1:y2, x1:x2, :3] = color
    return image


def random_color(image):
    h, w, _ = image.shape
    color = np.zeros((h, w, 3))
    color[:, :, 0] += random.randint(0, 255)
    color[:, :, 1] += random.randint(0, 255)
    color[:, :, 2] += random.randint(0, 255)
    image[:, :, :3] = color
    return image


def place_image_on_background(labels, image, background, coordinates, i):
    # Prepare and resize image
    image, labels = preprossesing_image(image, labels)

    # Prepare image.
    mask = get_img_mask(image)
    image[:, :, 3] = mask
    # Create random color for bboxes or for formula.
    image = random_color(image)  # random_color(image, labels)
    background_alpha = 1.0 - mask
    scale = 1

    # resize (not yet) and find coordinates
    x_start, y_start, x_end, y_end, flag = provide_random_coordinates(
        background, image, coordinates, i)

    if flag:
        return background, None, flag

    # use numpy indexing to place the resized image in the background image
    for c in range(0, 3):
        background[y_start:y_end, x_start:x_end, c] = \
            (mask * image[:, :, c] + background_alpha *
             background[y_start:y_end, x_start:x_end, c])

    label_text = create_label_objects(x_start, y_start, labels, scale)
    # plt.imshow(background)
    # plt.show()
    return background, label_text, flag
