'''
Description:
    Prepare the data for training and testing.
'''
import numpy as np
import skimage.io
import tensorflow as tf
import scipy.io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from glob import glob
import matplotlib.pyplot as plt
import random
import os

# ==========================================================
#                      !!! WARNING !!!
#
# For now, I just take images from 2008 and
# corresponding annotations for model training.
# You may want to change the code when you want more data.
# ==========================================================

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 32
IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 151

@tf.function
def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and return
    a dictionary.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)

    mask = tf.image.decode_png(mask, channels=1)

    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

    return {'image': image, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if random.random() > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation."""
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def prepare_data(dataset_path):
    TRAINSET_SIZE = len(glob(dataset_path + "training/" + "*.jpg"))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    
    VALSET_SIZE = len(glob(dataset_path + "validation/" + "*.jpg"))
    print(f"The Validation Dataset contains {VALSET_SIZE} images.")
    
    train_dataset = tf.data.Dataset.list_files(dataset_path + "training/" + "*.jpg", seed=42)
    train_dataset = train_dataset.map(parse_image)
    
    val_dataset = tf.data.Dataset.list_files(dataset_path + "validation/" + "*.jpg", seed=42)
    val_dataset =val_dataset.map(parse_image)

    dataset = {"train": train_dataset, "val": val_dataset}

    # -- Train Dataset --#
    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=42)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    #-- Validation Dataset --#
    dataset['val'] = dataset['val'].map(load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    return dataset, TRAINSET_SIZE, VALSET_SIZE, BATCH_SIZE

def display_sample(display_list):
    plt.figure(figsize=(9, 9))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    dataset_path = "./data/ADEChallengeData2016/images/"
    dataset, _, _= prepare_data(dataset_path)
    print(dataset['train'])
    print(dataset['val'])
    for image, mask in dataset['train'].take(1):
        display_sample([image[0], mask[0]])
