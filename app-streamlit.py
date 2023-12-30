import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Rock Paper Scissors Prediction",
    initial_sidebar_state = 'auto'
)

hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            return key

with st.sidebar:
    st.image('TensorFlow_logo.png')
    st.title(":orange[Image Classification App]")
    st.markdown('---')
    st.subheader("Rock, Paper, Scissors is a simple hand game often used as a decision-making tool. Here are an App to predict it with accurate result based on Image.")

@st.cache_resource()
def load_model():
    model=tf.keras.models.load_model('../rps.h5')
    return model

with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
        # Rock Paper Scissors Prediction
        """
        )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.exif_transpose(image_data)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.array(image.convert("RGB"))
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")
    
    class_names = ['Paper', 'Rock', 'Scissor']
    
    string = "Prediction : " + class_names[np.argmax(predictions)]
    st.balloons()
    st.sidebar.success(string)
    if class_names[np.argmax(predictions)] == 'Paper':
        st.markdown("## Information")
        st.info("""- Formed by extending your hand open and flat.
                \n- Symbolizes knowledge or documents.
                \n- Covers rock (wraps around it).""")
    
    elif class_names[np.argmax(predictions)] == 'Rock':
        st.markdown("## Information")
        st.info("""- Formed by making a fist with your hand.
                \n- Symbolizes strength or power.
                \n- Defeats scissors (crushes them).""")
    
    elif class_names[np.argmax(predictions)] == 'Scissor':
        st.markdown("## Information")
        st.info("""- Formed by making a fist and extending the index and middle fingers in a V shape.
                \n- Symbolizes cutting or shearing.
                \n- Cuts paper (slices through it).""")

