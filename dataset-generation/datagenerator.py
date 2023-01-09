import helper
import cv2
import random
import torch
import sys
from pathlib import Path
from tqdm import tqdm


def create_dataset_random(dataset_size, number_of_formulas):
    dataset_path = Path("archive") / f"dataset{dataset_size}"
    if not dataset_path.exists():
        dataset_path.mkdir()

    images_path = Path("archive") / f"dataset{dataset_size}" / "images"
    if not images_path.exists():
        images_path.mkdir()

    labels_path = Path("archive") / f"dataset{dataset_size}" / "labels"
    if not labels_path.exists():
        labels_path.mkdir()

    background_path = Path("archive") / "background_images"
    formulas_path = Path("archive") / "formulas"

    backgrounds = [cv2.imread(str(file)) for file in background_path.glob("*.jpg")]
    formulas_label = open(formulas_path / "label.txt", 'r')
    formulas_label = formulas_label.readlines()

    # Label dictionary encoding for Yolov7
    label_dict = {}
    num_distinct_labels = 0
    for j in tqdm(range(dataset_size), total=dataset_size):
        for background in backgrounds:
            coordinates = torch.zeros((number_of_formulas, 4))
            label_txt = open(labels_path / f"{j}.txt", 'a')
            for i in range(number_of_formulas):
                formula_label_int = random.randint(0, len(formulas_label)-1)
                formula_label = formulas_label[formula_label_int].split(' ')
                image = cv2.imread(formula_label[0])
                formula_label[-1] = formula_label[-1][:-1]
                labels = list(map(lambda x: x.split(','), formula_label[1:]))
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
                    for line_ in label_line:
                        # label_line format is [class, x_center, y_center, width, height]
                        label_txt.write(f"{line_[0]} {line_[1]} {line_[2]} {line_[3]} {line_[4]}")
                        # new line
                        label_txt.write("\n")
            cv2.imwrite(str(images_path / f"{j}.jpg"), background)
            label_txt.close()

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