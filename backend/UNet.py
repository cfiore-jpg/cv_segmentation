'''
Description:
    Define the UNet architecture.
'''
import requests
from tqdm import tqdm
import IPython.display as display
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
import tensorflow_addons as tfa
from PrepareData import prepare_data, display_sample

def build_UNET(img_size, n_channels, n_classes):
    dropout_rate = 0.5
    initializer = 'he_normal'
    input_size = (img_size, img_size, n_channels)
    
    inputs = Input(shape=input_size)
    
    #Encoder---------------------------------
    x = Conv2D(64, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #----------------------------------------
    
    x = Conv2D(512, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(512, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    #Decoder---------------------------------
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #----------------------------------------
    
    x = Conv2D(n_classes, 1, 1, padding="valid")(x)
    output = Activation("softmax")(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = output)
    model.compile(optimizer=tfa.optimizers.Adadelta, loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    model.summary()
    return model

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def show_predictions(dataset, model, num=1):
    """Show a sample prediction."""
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display_sample([image[0], mask[0], create_mask(pred_mask)[0]])

if __name__ == '__main__':
    dataset_path = "./data/ADEChallengeData2016/images/"
    dataset, _, _ = prepare_data(dataset_path)
    model = build_UNET(128, 3, 151)
    show_predictions(dataset['train'], model, num=2)
