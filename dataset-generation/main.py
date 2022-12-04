import helper
import os
import glob

def create_dataset(dataset=1000, img=30):
    images_path = "archive/extracted_images"
    background_path = "archive/background_images"
    labels = os.listdir(images_path)
    backgrounds = [file for file in glob.glob(background_path)]




if __name__ == "__main__":
    create_dataset()