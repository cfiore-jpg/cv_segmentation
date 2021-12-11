'''
Description:
    Define the SegNet architecture.
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

def build_SEGNET(img_size, n_channels, n_classes):
    dropout_rate = 0.5
    initializer = 'he_normal'
    input_size = (img_size, img_size, n_channels)
    # -- Encoder -- #
    # Block encoder 1
    inputs = Input(shape=input_size)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    conv_enc_1 = BatchNormalization()(conv_enc_1)
    conv_enc_1 = ReLU()(conv_enc_1)
    conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)
    conv_enc_1 = BatchNormalization()(conv_enc_1)
    conv_enc_1 = ReLU()(conv_enc_1)
    conv_enc_1 = MaxPooling2D()(conv_enc_1)
    # Block encoder 2
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_1)
    conv_enc_2 = BatchNormalization()(conv_enc_2)
    conv_enc_2 = ReLU()(conv_enc_2)
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_2)
    conv_enc_2 = BatchNormalization()(conv_enc_2)
    conv_enc_2 = ReLU()(conv_enc_2)
    conv_enc_2 = MaxPooling2D()(conv_enc_2)
    # Block encoder 3
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_2)
    conv_enc_3 = BatchNormalization()(conv_enc_3)
    conv_enc_3 = ReLU()(conv_enc_3)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)
    conv_enc_3 = BatchNormalization()(conv_enc_3)
    conv_enc_3 = ReLU()(conv_enc_3)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)
    conv_enc_3 = BatchNormalization()(conv_enc_3)
    conv_enc_3 = ReLU()(conv_enc_3)
    conv_enc_3 = MaxPooling2D()(conv_enc_3)
    # Block encoder 4
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)
    conv_enc_4 = BatchNormalization()(conv_enc_4)
    conv_enc_4 = ReLU()(conv_enc_4)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_4)
    conv_enc_4 = BatchNormalization()(conv_enc_4)
    conv_enc_4 = ReLU()(conv_enc_4)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_4)
    conv_enc_4 = BatchNormalization()(conv_enc_4)
    conv_enc_4 = ReLU()(conv_enc_4)
    conv_enc_4 = MaxPooling2D()(conv_enc_4)
    # Block encoder 5
    conv_enc_5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_4)
    conv_enc_5 = BatchNormalization()(conv_enc_5)
    conv_enc_5 = ReLU()(conv_enc_5)
    conv_enc_5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_5)
    conv_enc_5 = BatchNormalization()(conv_enc_5)
    conv_enc_5 = ReLU()(conv_enc_5)
    conv_enc_5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_5)
    conv_enc_5 = BatchNormalization()(conv_enc_5)
    conv_enc_5 = ReLU()(conv_enc_5)
    conv_enc_5 = MaxPooling2D()(conv_enc_5)
    # -- Decoder -- #
    # Block decoder 1
    conv_dec_1 = UpSampling2D()(conv_enc_5)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)
    conv_dec_1 = BatchNormalization()(conv_dec_1)
    conv_dec_1 = ReLU()(conv_dec_1)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)
    conv_dec_1 = BatchNormalization()(conv_dec_1)
    conv_dec_1 = ReLU()(conv_dec_1)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)
    conv_dec_1 = BatchNormalization()(conv_dec_1)
    conv_dec_1 = ReLU()(conv_dec_1)
    # Block decoder 2
    conv_dec_2 = UpSampling2D()(conv_dec_1)
    conv_dec_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)
    conv_dec_2 = BatchNormalization()(conv_dec_2)
    conv_dec_2 = ReLU()(conv_dec_2)
    conv_dec_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)
    conv_dec_2 = BatchNormalization()(conv_dec_2)
    conv_dec_2 = ReLU()(conv_dec_2)
    conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)
    conv_dec_2 = BatchNormalization()(conv_dec_2)
    conv_dec_2 = ReLU()(conv_dec_2)
    # Block decoder 3
    conv_dec_3 = UpSampling2D()(conv_dec_2)
    conv_dec_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)
    conv_dec_3 = BatchNormalization()(conv_dec_3)
    conv_dec_3 = ReLU()(conv_dec_3)
    conv_dec_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)
    conv_dec_3 = BatchNormalization()(conv_dec_3)
    conv_dec_3 = ReLU()(conv_dec_3)
    conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)
    conv_dec_3 = BatchNormalization()(conv_dec_3)
    conv_dec_3 = ReLU()(conv_dec_3)
    # Block decoder 4
    conv_dec_4 = UpSampling2D()(conv_dec_3)
    conv_dec_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
    conv_dec_4 = BatchNormalization()(conv_dec_4)
    conv_dec_4 = ReLU()(conv_dec_4)
    conv_dec_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
    conv_dec_4 = BatchNormalization()(conv_dec_4)
    conv_dec_4 = ReLU()(conv_dec_4)
    # Block decoder 5
    conv_dec_5 = UpSampling2D()(conv_dec_4)
    conv_dec_5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_5)
    conv_dec_5 = BatchNormalization()(conv_dec_5)
    conv_dec_5 = ReLU()(conv_dec_5)
    conv_dec_5 = Conv2D(n_classes, 1, 1, activation='relu', padding='valid', kernel_initializer=initializer)(conv_dec_5)
    conv_dec_5 = BatchNormalization()(conv_dec_5)
    output = Softmax()(conv_dec_5)
    # -- Build Model -- #
    model = tf.keras.Model(inputs = inputs, outputs = output)
    model.compile(optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3), loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
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
    dataset, _, _, _ = prepare_data(dataset_path)
    model = build_SEGNET(128, 3, 151)
    show_predictions(dataset['train'], model, num=2)
