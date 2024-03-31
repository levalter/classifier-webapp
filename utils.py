import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import torch
from torchvision import transforms


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
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