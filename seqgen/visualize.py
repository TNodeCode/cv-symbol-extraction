import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import string
import random


def load_image(symbol, class_samples, img_dir, erode=False):
    n_samples = len(class_samples[symbol])
    idx = random.randint(0, n_samples-1)
    # Read image.
    img = cv2.imread(f'{img_dir}/{symbol}/{class_samples[symbol][idx]}', cv2.IMREAD_GRAYSCALE)
    if erode:
        img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=1)
    return img


def map_symbols_to_classes(symbol):
    #mappings = {'a':'A', 'b':'B', 'c':'C', 'COMMA':',', 'd':'D', 'e':'E', 'f':'F', 'g':'G', 'h':'H', 'i':'I', 'j':'J', 'k':'K', 'l':'L', 'm':'M', 'n':'N', 'o':'O', 'p':'P', 'phi':'phi_lower', 'prime':',', 'q':'Q', 'r':'R', 's':'S', 't':'T', 'sum':'sigma_upper', 'U':'u', 'V':'v', 'W':'w', 'x':'X', 'Y':'y', 'Z':'z', '|': 'vert', '/':'vert'}
    mappings = {'COMMA':',', 'prime':',', 'sum':'sigma_upper', '|': 'vert', '/':'div'}
    if symbol in mappings.keys():
        return mappings[symbol]
    elif symbol in string.ascii_lowercase:
        return symbol + "_low"
    elif symbol in string.ascii_uppercase:
        return symbol.lower() + "_up"
    return symbol


def create_img_array_from_coordinates(coordinates, img_height, img_width, max_length):
    img_array = torch.zeros((img_height, img_width))
    for i in range(max_length):
        x0, y0, x1, y1 = coordinates[i].to(torch.int64)
        img_array[int(y0):int(y1), int(x0):int(x0)+1] = 1.0
        img_array[int(y0):int(y1), int(x1):int(x1)+1] = 1.0
        img_array[int(y0):int(y0)+1, int(x0):int(x1)] = 1.0
        img_array[int(y1):int(y1)+1, int(x0):int(x1)] = 1.0
    return img_array


def create_cv2img_array_from_coordinates(input_seqs, coordinates, img_height, img_width, class_samples, vocab, max_length, img_dir, smallest_index=3):
    img_array = np.ones((img_height, img_width))
    for i in range(max_length):
        if input_seqs[i] < smallest_index:
            continue
        symbol = vocab.idx2word[int(input_seqs[i])]
        img = load_image(map_symbols_to_classes(symbol.replace('\\', '')), class_samples, img_dir)
        if img is None:
            raise Exception(f"Image could not be generated")
        img = img / 255
        x0, y0, x1, y1 = coordinates[i].to(torch.int64)
        w, h = int(x1 - x0), int(y1 - y0)
        if (w > 0 and h > 0):
            img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
            window = img_array[int(y0):int(y0)+h, int(x0):int(x0)+w]
            img_array[int(y0):int(y0)+h, int(x0):int(x0)+w][window == 1] = img[window == 1]
    return img_array


def create_images_from_input_seqs(input_seqs, coordinates, batch_size, img_height, img_width, class_samples, vocab_in, max_length, img_dir):
    images = []
    for i in range(batch_size):
        images.append(
            np.expand_dims(
                create_cv2img_array_from_coordinates(
                    input_seqs[i],
                    coordinates[i],
                    img_height,
                    img_width,
                    class_samples,
                    vocab_in,
                    max_length,
                    img_dir
                ),
                axis=0
            )
        )
    return images


def create_image_patches(image_tensor, patch_size, flatten_patches=True):
    batch_size=image_tensor.shape[0]
    img_shape_dim0 = image_tensor.shape[2]
    img_shape_dim1 = image_tensor.shape[3]
    n_patches_dim0 = img_shape_dim0 // patch_size
    n_patches_dim1 = img_shape_dim1 // patch_size
    n_channels = image_tensor.shape[1]
    n_patches = n_patches_dim0 * n_patches_dim1 * n_channels
    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).reshape(batch_size, n_patches, patch_size*patch_size)
    if (not flatten_patches):
        return patches.reshape(batch_size, n_patches, patch_size, patch_size)
    return patches