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
    x1 = bbox[1] + x_start - 1
    y1 = bbox[0] + y_start - 1
    x2 = bbox[3] + x_start + 1
    y2 = bbox[2] + y_start + 1

    # # draw bbox on the image
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

def create_label_objects(x1_start, y1_start, labels, min_y, scale):
    label_txt = ''
    for bbox in labels:
        x1 = int(int(bbox[0]) * scale) + x1_start
        y1 = int((int(bbox[1]) - min_y) * scale) + y1_start
        x2 = int(int(bbox[2]) * scale) + x1_start + 2
        y2 = int((int(bbox[3]) - min_y) * scale) + y1_start + 2
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

def get_formulars_mask(formulas):
    """ Creates mask of given formula image. A mask will allow us to focus on the specific portion of the input formula image,
    in our case - a formula.

    :param formula: Input formula image.
    :return: Mask of image.
    """
    mask = np.copy(formulas)[:, :, :, 0]
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
    return img, scale

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
    background, _ = resize(background, 700)

    # Prepare image.
    image = swap_black_white(image)
    mask = get_img_mask(image)
    nz = np.nonzero(mask)
    min_y = np.min(nz[0])
    max_y = np.max(nz[0])
    image = image[min_y:max_y]
    img = image
    scale = 1


    # resize and find coordinates
    # rand_int = random.randint(int(image.shape[0]/2), int(image.shape[0]*1.3))
    # img, scale = resize(image, rand_int)
    co = torch.tensor(provide_random_coordinates(
        background, img)).reshape((1, 4))
    while not overlap(co, coordinates):
        # rand_int = random.randint(int(image.shape[0]/2), int(image.shape[0]*1.3))
        # img, scale = resize(image, rand_int)
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

    x1_start, y1_start, _, _ = create_bounding_box(mask, x_start, y_start)
    label_text = create_label_objects(x1_start, y1_start, label, min_y, scale)
    # plt.imshow(background)
    # plt.show()
    return background, label_text











###################################################################
def place_images_on_grid_background2(labels, images, background):
    # Prepare and resize background
    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    background = resize(background, 450)


    dims_x = int(images[0].shape[0]*1.3) # minimum width of grid cell cause of img shape
    dims_y = int(images[0].shape[1]*1.3) # minimum height of grid cell cause of img shape
    # Get grid depending on background image
    grid = random.randint(1, int(background.shape[0] / dims_x)), \
           random.randint(1, int(background.shape[1] / dims_y))

    dims_x = int(background.shape[0] / grid[0]) # resize to new dimensions
    dims_y = int(background.shape[1] / grid[1])

    for i in range(grid[0]):
        for j in range(grid[1]):
            rand_int = random.randint(0, len(images)-1)
            image = swap_black_white(images[rand_int])
            # resize and find coordinates
            img = resize(image, random.randint(
                int(image.shape[0]/2), int(image.shape[0]*1.3)))
            co = [i * dims_x, j * dims_y, (i * dims_x + img.shape[0]),
                  (j * dims_y + img.shape[1])]
            co = torch.tensor(co).reshape((1, 4))

            mask = get_img_mask(img)
            img[:, :, 3] = mask
            img = random_color(img)
            background_alpha = 1.0 - mask

            # get placement coordinates
            x_start, y_start, x_end, y_end = co[0].tolist()

            # use numpy indexing to place the resized image in the background image
            for c in range(0, 3):
                background[x_start:x_end, y_start:y_end, c] = \
                    (mask * img[:, :, c] + background_alpha *
                     background[x_start:x_end, y_start:y_end, c])

            x1, y1, x2, y2 = create_bounding_box(mask, x_start, y_start)
            label_text = create_label_object(x1, y1, x2, y2, labels[rand_int])
    return background, label_text


def place_images_on_grid_background(labels, images, background):
    # Prepare and resize background
    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    background = resize(background, 450)

    max_line_height = int(images[0].shape[1] * 1.3)

    # Get max amount of lines
    max_lines = int(background.shape[0] / max_line_height)

    # Place images on line
    for i in range(max_lines):
        image = swap_black_white(images[i])

        # get bounding box of image




