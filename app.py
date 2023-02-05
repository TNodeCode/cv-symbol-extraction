import streamlit as st
import torch
from PIL import Image
from app.seq2seq import predict_sequentially
import cv2
from yolov7 import run_yolo
import numpy as np
import os

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
    img = cv2.resize(img, (640, 640))

    # make chalk lines better
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('image.jpg', img)

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
    input_seq = torch.tensor([[0, 5, 5, 1, 1, 1]])
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.4, 0.3, 0.7, 0.7], [0.8, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    print("IN", input_seq.shape, "COORDS", coordinates.shape)
    prediction = predict_sequentially(
        input_seqs=input_seq, coordinates=coordinates)
    prediction = list(prediction)

    st.markdown("### Step 2: Generate LaTeX formula with Seq2Seq model")
    generated_formula = "a^2+b^2=c^2"
    st.write("This is the formula the model has generated")
    st.write("Raw LaTeX formula")
    st.code("".join(prediction))
    st.write("Parsed LaTeX formula")
    st.latex("".join(prediction))
