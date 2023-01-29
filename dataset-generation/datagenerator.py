import helper
import cv2
import random
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

# CUDA_VISIBLE_DEVICES=0,1

def _create_dataset(dataset_size, backgrounds, number_of_formulas, labels_path, formulas_label_dict, label_dict,
                    num_distinct_labels, images_path, formulas):
    # label_txt = open(labels_path / "label.txt", 'a')  # FÜR TILO
    amount_of_formulas = len(formulas)
    formulas_used = 0
    for j in tqdm(range(dataset_size), total=dataset_size):

        back = backgrounds[j % len(backgrounds)]
        coordinates = torch.zeros((number_of_formulas, 4))
        label_txt = open(labels_path / f"{j}.txt", 'a')
        background = helper.preprossesing_background(back)

        for i in range(number_of_formulas):
            image_path = formulas[formulas_used % amount_of_formulas]
            image = cv2.imread(str(image_path))
            formulas_used += 1
            # latex = formulas_label_dict[image_path.name.split('.')[0]] # FÜR TILO

            labels = list(map(lambda x: x.split(','), formulas_label_dict[image_path.name])) # FÜR TILO #2

            for label in labels:
                # check if label already in label_dict
                if label[-1] not in label_dict:
                    label_dict[label[-1]] = num_distinct_labels
                    num_distinct_labels += 1
                # Replace all "labels" with their corresponding label_dict value (aka the encoded yolov7 labels)
                label[-1] = label_dict[label[-1]]

            background, label_line, flag = helper.place_image_on_background(
                labels, image, background, coordinates, i)

            if not flag:
                # label_txt.write(latex) # FÜR TILO
                for line_ in label_line:
                    # label_line format is [class, x_center, y_center, width, height]
                    # label_txt.write(" ")  # FÜR TILO
                    label_txt.write(f"{line_[0]} {line_[1]} {line_[2]} {line_[3]} {line_[4]}")
                    # new line
                    label_txt.write("\n")
            # label_txt.write("\n") # FÜR TILO
        cv2.imwrite(str(images_path / f"{j}.jpg"), background)
        label_txt.close()
    return label_dict, num_distinct_labels
    # label_txt.close() # FÜR TILO


def create_dataset_random(dataset_size, number_of_formulas):
    random.seed(0)
    dataset_path = Path("archive") / f"dataset{dataset_size}"
    if not dataset_path.exists():
        dataset_path.mkdir()

    train_images_path = Path("archive") / f"dataset{dataset_size}" / "train" / "images"
    if not train_images_path.exists():
        train_images_path.mkdir(parents=True)

    train_labels_path = Path("archive") / f"dataset{dataset_size}" / "train" / "labels"
    if not train_labels_path.exists():
        train_labels_path.mkdir(parents=True)

    val_images_path = Path("archive") / f"dataset{dataset_size}" / "val" / "images"
    if not val_images_path.exists():
        val_images_path.mkdir(parents=True)

    val_labels_path = Path("archive") / f"dataset{dataset_size}" / "val" / "labels"
    if not val_labels_path.exists():
        val_labels_path.mkdir(parents=True)

    background_path = Path("archive") / "background_images"
    formulas_path = Path("archive") / "formulas"

    backgrounds = [cv2.imread(str(file)) for file in background_path.glob("*.jpg")]
    formulas_label = open(formulas_path / "label.txt", 'r')
    formulas_label = formulas_label.readlines()

    # Create dict from formulas_label with key as path to image without archive/formulas/ and value the rest of the line
    formulas_label_dict = {}
    for line in formulas_label:
        line = line.rstrip('\n')
        line = line.split(' ')
        key = line[0].split(os.sep)[2]
        formulas_label_dict[key] = line[1:]

    formulas = [file for file in formulas_path.glob("*.png")]

    # Create train and validation and test sets of 80/20 split
    train_formulas, val_formulas = train_test_split(formulas, test_size=0.1, random_state=0)

    # Label dictionary encoding for Yolov7
    label_dict = {"formula": 0}
    num_distinct_labels = 1

    label_dict, num_distinct_labels = \
        _create_dataset(dataset_size=int(dataset_size*0.8), backgrounds=backgrounds, number_of_formulas=number_of_formulas,
                    labels_path=train_labels_path, formulas_label_dict=formulas_label_dict, label_dict=label_dict,
                    num_distinct_labels=num_distinct_labels, images_path=train_images_path, formulas=train_formulas)

    _create_dataset(dataset_size=int(dataset_size*0.2), backgrounds=backgrounds, number_of_formulas=number_of_formulas,
                    labels_path=val_labels_path, formulas_label_dict=formulas_label_dict, label_dict=label_dict,
                    num_distinct_labels=num_distinct_labels, images_path=val_images_path, formulas=val_formulas)

    with open(dataset_path / 'encoded.txt', 'w') as f:
        # Iterate through the dictionary
        for k, v in label_dict.items():
            # Write each key-value pair to the file, with a newline character at the end
            f.write(f"{v}: {k}\n")


def main():
    args = sys.argv[1:]
    dataset_num = int(args[0])
    img_num = int(args[1])
    create_dataset_random(dataset_num, img_num)


if __name__ == "__main__":
    main()