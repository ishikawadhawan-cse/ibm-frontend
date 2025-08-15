import streamlit as st
import numpy as np
import cv2
import kagglehub
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# HAM10000 Labels
HAM_CLASSES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]

@st.cache_resource
def load_model():
    """Load ResNet50 pretrained on ImageNet with new output layer for HAM10000 classes."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(len(HAM_CLASSES), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # NOTE: This is not trained on HAM10000; acts as placeholder
    return model

def prepare_image(img_bytes):
    """Preprocess uploaded image for prediction."""
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_image(model, img_bytes):
    """Predict probabilities for the image."""
    processed_img = prepare_image(img_bytes)
    preds = model.predict(processed_img)
    return dict(zip(HAM_CLASSES, preds[0]))

@st.cache_resource
def load_sample_images():
    """Download small sample of HAM10000 images."""
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    img_dir1 = os.path.join(path, "HAM10000_images_part_1")
    img_dir2 = os.path.join(path, "HAM10000_images_part_2")
    all_imgs = [os.path.join(img_dir1, f) for f in os.listdir(img_dir1) if f.endswith(".jpg")]
    all_imgs += [os.path.join(img_dir2, f) for f in os.listdir(img_dir2) if f.endswith(".jpg")]
    return random.sample(all_imgs, 5)
