## This web-app can recognise a random image uploaded by the User and classify it into one of the following 6 categories of Natural Scenes : [ 'Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street' ]

## Model Architecture : VGG19 

## Dataset : Intel Image Classification dataset | Published by Intel to host a image classification challenge, it consists of 25,000 coloured images of natural scenes around the world of size 150x150 pixels

Transfer Learning(VGG19) was implemented by using the pre-trained weights from ImageNet dataset. The last layer has been changed to classify the outputs into 6 categories and the model was trained again for 30 epochs with 
0.00001 learning rate and Adam optimizer. 

The model achieved 91.47 % accuracy on the validation dataset.

Data augmentation techniques like RandomHorizontalFlip, RandomRotation and ColorJitter were applied to the data. 
Before using them for training the model, the images were normalized with ImageNet stats and the pixels were converted to tensor.

## Web-App

The weights of the saved model was downloaded and an we application was built using Streamlit for a more accessible and intuitive user interface.
Still, the model could only be accessed from my local machine.

## Deployment

Finally, the web app was deployed on a public URL using Heroku so that it could be accessed by any person from any device.
