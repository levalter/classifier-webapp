import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import torch
from torchvision import transforms


def set_page_bg(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    return


def predict(image, model, labels):
    transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    image = transform(image).view(1, 3, 224, 224)
    pred  = model.forward(image)
    prob, idx = torch.max(torch.sigmoid(pred), dim=1)
    prob = prob.detach().numpy()[0]
    idx = idx.numpy()[0]
    return labels[idx], prob