import streamlit as st
import torch
from io import StringIO
from PIL import Image
from app.seq2seq import model, predict_sequentially

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

    '''
    TODO run the image through the YOLO model
    '''

    st.markdown("### Step 1: Detect symbols with YOLO")
    image_yolo = Image.open('image.jpg')
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
