import streamlit as st
from PIL import Image, ImageFilter

import torchvision.transforms as transforms
from torchvision import *
from torch import *
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
      super().__init__()
      self.pretrained_model = models.resnet18(weights='DEFAULT')
      
      feature_in = self.pretrained_model.fc.in_features
      self.pretrained_model.fc = torch.nn.Linear(feature_in, 42)

    def forward(self,x):
      return self.pretrained_model(x)


st.write('''<style>
            body{
            text-align:center;
            background-color:#ACDDDE;

            }

            </style>''', unsafe_allow_html=True)

st.title('Central Asian Food Detector')


transform = transforms.Compose([transforms.Resize((380,380)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


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
    image = transform(image).view(1,3,380,380)

    pred  = model.forward(image)
    prob, idx = torch.max(torch.sigmoid(pred), dim = 1)
    prob = prob.detach().numpy()[0]
    idx = idx.numpy()[0]    

    st.text(f'class {labels[idx]}')
    st.write('confidence {:0.3f}'.format(float(prob)))

