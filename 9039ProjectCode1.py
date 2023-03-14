import numpy as np
import pandas as pd
import numpy as np
import glob as gb
import os
import seaborn as sns
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
# Enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_df = pd.read_csv('archive/train.csv', sep=" ", header=None)
train_df.columns = ['patient id', 'filename', 'label', 'data source']
train_df = train_df.drop(['patient id', 'data source'], axis=1)
train_df.head()


test_df = pd.read_csv('archive/test.csv', sep=" ", header=None)
test_df.columns = ['patient id', 'filename', 'label', 'data source']
test_df = test_df.drop(['patient id', 'data source'], axis=1)
test_df.head()


train_p_count = train_df['label'].value_counts()['positive']
train_n_count = train_df['label'].value_counts()['negative']

print(train_df['label'].value_counts())
print(test_df['label'].value_counts())
train_df = shuffle(train_df)


train_folder_path = 'archive/train'
test_folder_path = 'archive/test'

# Print out each folder's info


def count_photos(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            count += 1
    return count


def count_photos_in_folders(train_folder_path, test_folder_path):
    train_count = count_photos(train_folder_path)
    test_count = count_photos(test_folder_path)
    return (train_count, test_count)


# 将数据信息打印出来
# 2. Print out info for each dataset
train_count, test_count = count_photos_in_folders(
    train_folder_path, test_folder_path)

print(f"Train folder contains {train_count} photos.")
print(f"Test folder contains {test_count} photos.")

train_data_size = [train_p_count, train_n_count]
labels = ['positive', 'negative']

fig, axs = plt.subplots(figsize=(12, 8))
sns.barplot(x=train_data_size, y=labels, ax=axs)
axs.set_title('Training data categories')
axs.set(xlabel='Image number')
plt.show()


# negative values in class column
negative = train_df[train_df['label'] == 'negative']
# positive values in class column
positive = train_df[train_df['label'] == 'positive']

df_majority_downsampled_neg = resample(negative, replace=True, n_samples=5000)
df_majority_downsampled_pos = resample(positive, replace=True, n_samples=5000)

train_df = pd.concat([df_majority_downsampled_pos,
                     df_majority_downsampled_neg])
train_df = shuffle(train_df)  # shuffling so that there is particular sequence


train_df, valid_df = train_test_split(train_df, train_size=0.8, random_state=0)

print(
    f"Negative and positive values of train:\n {train_df['label'].value_counts()}")
print(
    f"Negative and positive values of validation:\n {valid_df['label'].value_counts()}")
print(
    f"Negative and positive values of test:\n {test_df['label'].value_counts()}")


# 读取数据，把train分成train set和validation set，将batch size设置为64

train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# Now fit the them to get the images from directory (name of the images are given in dataframe) with augmentation


train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory='archive/train', x_col='filename',
                                              y_col='label', target_size=(256, 256), batch_size=64,
                                              class_mode='binary')
val_gen = test_datagen.flow_from_dataframe(dataframe=valid_df, directory='archive/train', x_col='filename',
                                           y_col='label', target_size=(256, 256), batch_size=64,
                                           class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(dataframe=test_df, directory='archive/test', x_col='filename',
                                            y_col='label', target_size=(256, 256), batch_size=64,
                                            class_mode='binary')
labels = {value: key for key, value in train_gen.class_indices.items()}
# class mode binary because we want the classifier to predict covid or not
# target size (200,200) means we want the images to resized to 200*200

# Examine the first image in the training dataset and print the label which corresponding to it.
print(labels)
images, labels = next(train_gen)
# Get the filenames associated with the images in the batch
filenames = train_gen.filenames
# Print the filename and label for the first image in the batch
print(filenames[0], labels[0])
# Plot the first image in the batch
plt.imshow(images[0])


##### 此部分开始建模 #####
# Modeling
# Define the CNN model
CNNmodel = models.Sequential()
CNNmodel.add(layers.Conv2D(
    32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
CNNmodel.add(layers.MaxPooling2D((2, 2)))

# Dropout rate was added to prevent overfitting.
CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(layers.MaxPooling2D((2, 2)))

CNNmodel.add(layers.Conv2D(128, (3, 3), activation='relu'))
CNNmodel.add(layers.MaxPooling2D((2, 2)))
CNNmodel.add(layers.Dropout(0.25))

CNNmodel.add(layers.Conv2D(256, (3, 3), activation='relu'))
CNNmodel.add(layers.MaxPooling2D((2, 2)))
CNNmodel.add(layers.Dropout(0.25))

CNNmodel.add(Flatten())
CNNmodel.add(layers.Dense(64, activation='sigmoid'))
CNNmodel.add(layers.Dropout(0.25))
CNNmodel.add(layers.Dense(1, activation='sigmoid'))
CNNmodel.summary()
# Compile the model
# implement the early stopping technique to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5)
CNNmodel.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])


# Train the model
history = CNNmodel.fit(train_gen,
                       #steps_per_epoch=train_gen.samples // train_gen.batch_size,
                       epochs=20,
                       validation_data=val_gen,
                       #validation_steps=val_gen.samples // val_gen.batch_size,
                       callbacks=[early_stop])

# Evaluate the model on the test set
test_loss, test_acc = CNNmodel.evaluate(
    test_gen, steps=test_gen.samples // test_gen.batch_size)


# Evaluate the model on the test set
test_loss, test_acc = CNNmodel.evaluate(
    test_gen, steps=test_gen.samples // test_gen.batch_size)


fig, ax = plt.subplots(nrows=2, figsize=(12, 10))
ax[0].set_title('Loss vs Epoch')
ax[0].plot(history.history['loss'], label='Training Loss', color='orange')
ax[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='best')
ax[1].set_title('Accuracy vs Epoch')
ax[1].plot(history.history['accuracy'],
           label='Training Accuracy', color='orange')
ax[1].plot(history.history['val_accuracy'],
           label='Validation Accuracy', color='red')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend(loc='best')
plt.tight_layout()
plt.show()
print('Test accuracy:', test_acc)
