import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

model = keras.models.load_model('pizza_model.h5')

img = image.load_img('pizza.jpg', target_size=(224, 224))

x = image.img_to_array(img)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)

if prediction[0][0] > 0.7:
    print(f'pizza ({prediction})')
else:
    print(f'no_pizza ({prediction})')
