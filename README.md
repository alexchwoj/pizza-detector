# Pizza Detector
Artificial intelligence model to detect if an image contains a pizza or not

## Overview

This model uses convolutional neural networks (CNNs) to classify images as either pizza or not pizza. It was built using the Keras API in TensorFlow. The model takes input images of size 224x224x3 and predicts a binary output of either 0 (not pizza) or 1 (pizza).

The model was trained on a dataset of images that included both pizza and non-pizza images. The pizza images were obtained from various online sources and consisted of different types of pizza. The non-pizza images were randomly selected images of food that were not pizza.

## Dataset

The dataset was divided into a training set and a validation set. The training set contained 750 images (375 pizza and 375 non-pizza), while the validation set contained 150 images (75 pizza and 75 non-pizza). The images were preprocessed by rescaling the pixel values between 0 and 1 and using data augmentation techniques to increase the diversity of the training data.

## Model Architecture

The model architecture consists of three convolutional layers with max pooling layers in between, followed by two dense layers. The output layer has a sigmoid activation function to output a probability between 0 and 1.

The model was trained using binary crossentropy loss and the Adam optimizer with a learning rate of 0.001. The model was trained for 10 epochs with a batch size of 32.

## Performance

The model achieved an accuracy of 95% on the validation set, indicating that it is able to accurately classify pizza and non-pizza images. The model can be further improved by fine-tuning the hyperparameters, increasing the size of the dataset, or using more advanced techniques such as transfer learning.

## Usage

The trained model can be loaded and used to classify new images as either pizza or not pizza. The model takes as input an image of size 224x224x3 and outputs a probability of the image being pizza. The model can be easily integrated into a web or mobile application for real-time pizza detection.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

model = load_model('pizza_model.h5')

# load and preprocess the image
img = load_img('pizza.jpg', target_size=(224, 224))
img = img_to_array(img)
img = img.reshape(1, 224, 224, 3)
img = img.astype('float32')
img = img / 255.0

# make a prediction
pred = model.predict(img)

if pred > 0.5:
    print('Pizza')
else:
    print('Not pizza')

```
![enter image description here](https://media.discordapp.net/attachments/850593437396500500/1076304272775458957/photo_2023-02-17_20-29-34.jpg)
![enter image description here](https://media.discordapp.net/attachments/850593437396500500/1076304273048092703/photo_2023-02-17_20-29-34_2.jpg)