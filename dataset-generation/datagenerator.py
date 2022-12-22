import helper
import os
import cv2
import random
import torch
import sys
from pathlib import Path

def create_dataset_random(dataset_num, img_num):
    dataset_path = Path("archive") / f"dataset{dataset_num}"
    if not dataset_path.exists():
        dataset_path.mkdir()

    images_path = Path("archive") / "extracted_images"
    background_path = Path("archive") / "blackboard_images"
    labels = [f for f in os.listdir(images_path) if not f.startswith('.')]
    backgrounds = [cv2.imread(str(file)) for file in background_path.glob("*.jpg")]

    filename = 0
    label_txt = open(dataset_path / "label.txt", 'a')

    while filename < dataset_num:
        for back in backgrounds:
            coordinates = torch.zeros((img_num, 4))
            label_txt.write(str(Path.cwd() / dataset_path / f"{filename}.jpg "))
            background = back
            for i in range(img_num):
                label_int = random.randint(0, len(labels)-1)
                label = labels[label_int]
                image_list = list((images_path / label).glob("*.jpg"))
                image = cv2.imread(str(random.choice(image_list)))

                background, label_line = helper.place_image_on_background(
                    label, image, background, coordinates, i)

                label_txt.write(label_line)

            label_txt.write('\n')

            cv2.imwrite(str(dataset_path / f"{filename}.jpg"), background)

            filename += 1
            if filename >= dataset_num:
                break

    label_txt.close()

def create_dataset_grid(dataset_num, img_num):
    dataset_path = Path("archive") / f"dataset{dataset_num}"
    if not dataset_path.exists():
        dataset_path.mkdir()

    images_path = Path("archive") / "extracted_images"
    background_path = Path("archive") / "blackboard_images"
    labels = [f for f in os.listdir(images_path) if not f.startswith('.')]
    backgrounds = [cv2.imread(str(file)) for file in background_path.glob("*.jpg")]

    filename = 0
    label_txt = open(dataset_path / "label.txt", 'a')

    while filename < dataset_num:
        for back in backgrounds:
            img_list = []
            label_list = []
            label_txt.write(str(Path.cwd() / dataset_path / f"{filename}.jpg "))
            background = back
            for i in range(img_num):
                label_int = random.randint(0, len(labels)-1)
                label = labels[label_int]
                image_list = list((images_path / label).glob("*.jpg"))
                image = cv2.imread(str(random.choice(image_list)))
                img_list.append(image)
                label_list.append(label)

            background, label_line = helper.place_images_on_grid_background(
                label_list, img_list, background)

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


#if __name__ == "__main__":
#    main()

create_dataset_grid(2, 30)
