import os
import torch
import torchvision

import streamlit as st
from streamlit import file_uploader

import torch.nn as nn
import torchvision.transforms as tt

from torchvision import models
from PIL import Image
from torch import max 





CLASSES = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
IMAGENET_STATS = ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
DEVICE = 'cpu'



# Apply transformations to User uploaded image
test_transforms = tt.Compose([
    tt.Resize((150, 150)),
    tt.ToTensor(),
    tt.Normalize(*IMAGENET_STATS)
])






# Label Predictor Function
def predict_label(img, model):
    
    #img = Image.open(img_path)
    pimg = test_transforms(img).unsqueeze(0).to(DEVICE)
    prediction = model(pimg)
    _, tpredict = torch.max(prediction.data, 1)
    
    return CLASSES[tpredict[0].item()]




st.title("Natural Scene Recognition")

st.header("Multi-class Image Classification model using Transfer Learning( VGG19 )")
st.markdown("Here's the link to repo on " + "[GitHub](https://github.com/tishya-tripathi)") # !!! Change link to repo !!!


st.write(" ")

st.write("This web-app can classify Natural Scenes around the world into one of the following 6 categories: ")
st.markdown(
    """
    * Buildings
    * Forest
    * Glacier
    * Mountain
    * Sea
    * Street
    """
    )


st.write(" ")
st.write(" ")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])    # User uploads image file



if(st.button("PREDICT")):

    if file is None:

        st.error("Please upload an image file of type .jpeg/.jpg/.png  ! ")

    else:

        img = Image.open(file)

        st.image(img, use_column_width=True)

        # Work in progress
        with st.spinner('Work In Progress...   ;)'):

            # Load our model to CPU
            model = models.vgg19(pretrained=True).to(DEVICE)

            model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(CLASSES)).to(DEVICE)

            # Load our previously saved Weights from Model 

            checkpoint = torch.load(r"D:\VS Code Programs\Img Classifier Deployment using Streamlit\checkpoint_1.pth" , map_location=torch.device('cpu'))  # PATH to checkpoint_1.pth
            model.load_state_dict(checkpoint)

            pred = predict_label(img, model)
         
            st.balloons()
            st.success("Predicted Label :  **{}**".format( pred ) )

    

