import streamlit as st
from PIL import Image, ImageFilter

import torchvision.transforms as transforms
from torchvision import *
from torch import *
import torch.nn as nn
from utils import set_page_bg, predict

class net(nn.Module):
    def __init__(self):
      super().__init__()
      self.pretrained_model = models.resnet18(weights='DEFAULT')
      
      feature_in = self.pretrained_model.fc.in_features
      self.pretrained_model.fc = torch.nn.Linear(feature_in, 42)

    def forward(self,x):
      return self.pretrained_model(x)


set_page_bg('./backgrounds/bg3.png')


st.write('''<style>
            body{
            text-align:center;
            background-color:#ACDDDE;

            }

            </style>''', unsafe_allow_html=True)

st.title('Central Asian Food Detector')


# Load the model
model = net()
weights = torch.load('model/resnet18_final.pt', map_location=torch.device('cpu'))
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in weights.items():
    name = 'pretrained_model.' + k # add 'pretrained_model'
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

with open('./model/labels.txt', 'r') as f:
    labels = [a for a in f.readlines()]
    f.close()

st.text('Model loaded using resnet18')
file_type = ['jpg', 'jpeg', 'png']
uploaded_file = st.file_uploader("Choose a  file", type = file_type)


if uploaded_file != None:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.filter(ImageFilter.MedianFilter)
    st.image(image)
    label, conf = predict(image, model, labels) 

    st.text(f'class {label}')
    st.text('confidence {:0.3f}'.format(float(conf)))

