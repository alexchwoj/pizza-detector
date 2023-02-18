import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = "pizza_dataset"

data_dir + "/train"

def change_names(folder_path, prefix):
    for count, filename in enumerate(os.listdir(folder_path)):
        new_filename = f"{prefix}_{count}.jpg"

        current_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        try:
            os.rename(current_path, new_path)
        except:
            pass

change_names(data_dir + "/train/pizza/", "pizza")
change_names(data_dir + "/train/no_pizza/", "no_pizza")
change_names(data_dir + "/validation/pizza/", "pizza")
change_names(data_dir + "/validation/no_pizza/", "no_pizza")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir + "/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    data_dir + "/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=val_generator)
model.save('pizza_model.h5')