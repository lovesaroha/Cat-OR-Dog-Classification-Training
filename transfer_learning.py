# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model on cats and dogs images.
from tensorflow import keras
import numpy
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model

# Parameters.
epochs = 10
batchSize = 128

# Data url (https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip).
# Inception model weights url (https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5).

# Load training images from location and change image size.
training_data = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest').flow_from_directory(
    'cats_and_dogs_filtered/train',
    target_size=(150, 150),
    batch_size=batchSize,
    class_mode='binary')

# Load validation images from location and change image size.
validation_data = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest').flow_from_directory(
    'cats_and_dogs_filtered/validation',
    target_size=(150, 150),
    batch_size=batchSize,
    class_mode='binary')


# Trained model.
trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

trained_model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Set training false for trained model.
for layer in trained_model.layers:
  layer.trainable = False

# Last layer output.
last_layer_output = trained_model.get_layer('mixed7').output

# Create model with 1 output unit for classification.
x = layers.Flatten()(last_layer_output)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense  (1, activation="sigmoid")(x) 
model = Model(trained_model.input, x)

# Set loss function and optimizer.
model.compile(optimizer="adam",
              loss='binary_crossentropy', metrics=['accuracy'])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
model.fit(
      training_data,
      steps_per_epoch=8,  
      epochs=epochs,
      verbose=1,
      callbacks=[checkAccuracy] ,
      validation_data=validation_data ,
      validation_steps=8)

# Predict on a image.
file = image.load_img("cats_and_dogs_filtered/validation/dogs/dog.2000.jpg", target_size=(150, 150))
x = image.img_to_array(file)
x = numpy.expand_dims(x, axis=0)
image = numpy.vstack([x])

# Predict.
prediction = model.predict(image)
print(prediction)




