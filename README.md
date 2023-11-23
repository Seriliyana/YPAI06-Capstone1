# YPAI06-Capstone1
# Capstone 1
## CV Assignment 1
### Cracks or No Cracks?
1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications, models
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
from sklearn.model_selection import train_test_split
import shutil
2. Data Loading
dataset_concrete = r"D:\Yayasan Peneraju\CAPSTONE 1\Dataset"

PATH = r"D:\Yayasan Peneraju\CAPSTONE 1\Split_dataset"

# Define your destination directories
train_dir = r"D:\Yayasan Peneraju\CAPSTONE 1\Split_dataset\train"
validation_dir = r"D:\Yayasan Peneraju\CAPSTONE 1\Split_dataset\validation"
test_dir = r"D:\Yayasan Peneraju\CAPSTONE 1\Split_dataset\test"

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
3. Inspect some data examples
class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')
4. Further split validation and test dataset
test_batches = tf.data.experimental.cardinality(test_dataset)
val_dataset = test_dataset.take(test_batches//5)
test_dataset = test_dataset.skip(test_batches//5)
5. Convert the tensorflow datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size = AUTOTUNE)
validation_dataset = test_dataset.prefetch(buffer_size = AUTOTUNE)
6. Create a sequential model for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))
7. Repeatedly apply image augmentation on a single image
for image, _ in train_dataset.take(1):
    first_image = image[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,0))
        plt.imshow(augmented_image[0]/255)
        plt.axis('off')
        plt.grid('off')
8.Define a layer for the data Normalization
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
#(A) Load the pretrained model using keras.application
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.summary()
keras.utils.plot_model(base_model)
#(B) Freeze the entire feature extractor
base_model.trainable = False
base_model.summary()
#(C) Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#(D) Create the output layer
output_layer = layers.Dense(len(class_names),activation='softmax')
#(E)Build the entire pipeline using Functional API
#a. Input
inputs = keras.Input(shape =IMG_SHAPE)
#b. Data augmentation
x = data_augmentation(inputs)
#c. Data Normalization
x = preprocess_input(x)
#d. Transfer learning feature extractor
x = base_model(x, training = False)
#e. Classification layers
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)
#e. Build the Model
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()
9. Compile the Model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
10. Create a Tensorboard callback object
PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(logpath)

#Evaluate the model before training
model.evaluate(test_dataset)
11. Model Training
early_stopping = callbacks.EarlyStopping(patience = 2)
EPOCHS = 10
history = model.fit(train_dataset,validation_data=validation_dataset,epochs=EPOCHS,callbacks=[tb,early_stopping])
# Evaluate the model with test data
model.evaluate(test_dataset)
12. Fine tune the model
fine_tune_at = 100
#freeze all the layers before the 'fine_tune at' layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
base_model.summary()
13. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(train_dataset, validation_data=validation_dataset,epochs=total_epoch,initial_epoch=history.epoch[-1],callbacks=[tb,early_stopping])
#Evaluate the model
model.evaluate(test_dataset)
15. Deployment
#(A)Retrieve a batch of images from test data and perform prediction
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
#(B)Display results in matplotlib
prediction_indexes = np.argmax(predictions,axis=1)
#Create a label map for the classes
label_map = {i:names for i, names in enumerate(class_names)}
prediction_label = [label_map[i] for i in prediction_indexes]
label_class_list = [label_map[i] for i in label_batch]

plt.figure(figsize=(15,15))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(f"Label:{label_class_list[i]}, Prediction:{prediction_label[i]}")
    plt.axis('off')
    plt.grid('off')
model.save('Capstone_1_Concrete_Crack_Images_for_Classification.h5')
