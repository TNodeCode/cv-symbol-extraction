import helper
import os
import cv2
import random
import torch
import sys
from pathlib import Path
import numpy as np


def create_dataset_random(dataset_num, img_num):
    dataset_path = Path("archive") / f"dataset{dataset_num}"
    if not dataset_path.exists():
        dataset_path.mkdir()

    background_path = Path("archive") / "background_images"
    # images_path = Path("archive") / "extracted_images"
    formulas_path = Path("archive") / "formulas"

    # labels = [f for f in os.listdir(images_path) if not f.startswith('.')]
    backgrounds = [cv2.imread(str(file)) for file in background_path.glob("*.jpg")]
    # formulas = np.array([cv2.imread(str(file)) for file in formulas_path.glob("*.png")])
    formulas_label = open(formulas_path / "label.txt", 'r')
    formulas_label = formulas_label.readlines()


    filename = 0
    label_txt = open(dataset_path / "label.txt", 'a')

    while filename < dataset_num:
        for back in backgrounds:
            coordinates = torch.zeros((img_num, 4)).long()
            label_txt.write(str(Path.cwd() / dataset_path / f"{filename}.jpg "))
            background = helper.preprossesing_background(back)
            for i in range(img_num):
                # print()
                # print('Image: ', i)
                formula_label_int = random.randint(0, len(formulas_label)-1)
                formula_label = formulas_label[formula_label_int].split(' ')
                image = cv2.imread(formula_label[0])
                formula_label[-1] = formula_label[-1][:-1]
                label = list(map(lambda x: x.split(','), formula_label[1:]))

                # Und noch kein resozing

                background, label_line, flag = helper.place_image_on_background(
                    label, image, background, coordinates, i)
                
                if not flag:
                    label_txt.write(label_line)

            label_txt.write('\n')

            cv2.imwrite(str(dataset_path / f"{filename}.jpg"), background)

            filename += 1
            if filename >= dataset_num:
                break

    label_txt.close()


def main():
    args = sys.argv[1:]
    dataset_num = int(args[0])
    img_num = int(args[1])
    create_dataset_random(dataset_num, img_num)


if __name__ == "__main__":
    main()