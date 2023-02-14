import streamlit as st
import torch
from PIL import Image
from app.seq2seq import predict_sequentially
import cv2
from yolov7 import run_yolo
import numpy as np
import os
from pathlib import Path
from seqgen.preprocess import normalize_coordinates

st.title("Deep LaTex Formula Generator")

# Display an upload form for the image files
uploaded_image = st.file_uploader("Choose a file")

# handle uploaded image
if uploaded_image is not None:
    # To read file as bytes:
    bytes_data = uploaded_image.read()
    # save image file
    with open("image.jpg", "wb") as f:
        f.write(bytes_data)
    # load the image from the filesystem
    image = Image.open('image.jpg')
    # display the image
    st.image(image, caption='Raw image')

    # resize image to 640x640 with cv2
    img = cv2.imread('image.jpg')
    #img = cv2.resize(img, (640, 640))

    # make chalk lines better
    #kernel = np.ones((5, 5), np.uint8)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('image.jpg', img)

    # detect with yolov7
    img = run_yolo('image.jpg')

    st.markdown("### Step 1: Detect symbols with YOLO")
    image_yolo = Image.open('detections' + os.sep + 'image.jpg')
    st.write("Detected tokens")

    # Display image with bounding boxes from YOLO
    st.image(image_yolo, caption='Detected symbols')

    '''
    TODO run bounding boxes through Seq2Seq model
    '''
    # read in all txt files from detections/formulaLabels/img[:-4] in list
    print(img[:-4])
    txt_files = [file for file in Path('detections' + os.sep + 'formulaLabels' + os.sep + img[:-4]).glob("*.txt")]
    print(len(txt_files))
    st.markdown("### Step 2: Generate LaTeX formula with Seq2Seq model")
    for txt_file in txt_files:
        input_seq_other = []
        coordinates_other = []
        input_seq = []
        coordinates = []
        input_seq.append(0)
        coordinates.append([0, 0, 0, 0])
        # one txt file per formula
        # read in txt file
        labels = open(txt_file, "r").readlines()
        for label in labels:
            l = []
            label = label.strip('\n').split(" ")
            class_label = label[0]
            class_label = class_label.split(".")[0]
            class_label = int(class_label) + 3
            input_seq.append(int(class_label))
            l.append(float(label[1]))
            l.append(float(label[2]))
            l.append(float(label[3]))
            l.append(float(label[4]))
            coordinates.append(l)
        # transform input_seq to tensor
        counter = 0
        input_seq.append(1)
        while len(input_seq) < 25:
            input_seq.append(2)
            counter += 1

        input_seq_other.append(input_seq)

        input_seq = torch.Tensor(input_seq_other)

        input_seq = input_seq.to(torch.int64)

        #transform coordinates to tensor
        coordinates = np.array(coordinates)
        coordinates[:, 0] = coordinates[:, 0] - 0.5 * coordinates[:, 2]
        coordinates[:, 1] = coordinates[:, 1] - 0.5 * coordinates[:, 3]
        coordinates[:, 2] = coordinates[:, 0] + 0.5 * coordinates[:, 2]
        coordinates[:, 3] = coordinates[:, 1] + 0.5 * coordinates[:, 3]

        coordinates = np.array(normalize_coordinates(np.array([coordinates]), contains_class=False)).squeeze()
        coordinates = coordinates.tolist()
        coordinates_other.append(coordinates)
        coords_end = torch.Tensor([[0.0, 0.0, 0.0, 0.0] for i in range(counter+1)])
        coordinates_other = torch.tensor(coordinates_other)
        # add coords_end to coordinates_other
        coords_end = coords_end.unsqueeze(0)
        coordinates = torch.cat((coordinates_other, coords_end), dim=1)
        print("IN", input_seq.shape, "COORDS", coordinates.shape)
        prediction = predict_sequentially(
            input_seqs=input_seq, coordinates=coordinates)
        prediction = list(prediction)
        # remove all "<end>" and "<start>" and "<unk>" tokens
        prediction = [x for x in prediction if x != "<end>" and x != "<start>" and x != "<unk>"]
        generated_formula = "a^2+b^2=c^2"
        st.write("This is the formula the model has generated")
        st.write("Raw LaTeX formula")
        st.code("".join(prediction))
        st.write("Parsed LaTeX formula")
        st.latex("".join(prediction))

