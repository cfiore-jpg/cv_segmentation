'''
Description:
    Train and evaluate the model.
'''
from glob import glob
import shutil
import argparse
import zipfile
import hashlib
import requests
from tqdm import tqdm
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
import tensorflow_addons as tfa
from UNet import show_predictions, build_UNET
from PrepareData import prepare_data

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, model):
        super(DisplayCallback, self).__init__()
        self.dataset = dataset
        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.dataset, self.model)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def train(model, dataset_path, load=False):
    print("Gathering the data...", end=' ')
    dataset, TRAINSET_SIZE, VALSET_SIZE, BATCH_SIZE = prepare_data(dataset_path)
    print("done")
    STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE
    EPOCHS = 20
    
    logdir = os.path.join("./backend/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    callbacks = [
    DisplayCallback(dataset['train'], model),
    tensorboard_callback,
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('./backend/best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)]
    
    if load:
        print("Loading model...", end=' ')
        model.load_weights('./backend/best_model_unet.h5')
        print("done")
        
    print("Let's train...")
    model_history = model.fit(dataset['train'], epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=dataset['val'],
                    callbacks=callbacks)

if __name__ == '__main__':
    dataset_path = "./data/ADEChallengeData2016/images/"
    model = build_UNET(128, 3, 151)
    train(model, dataset_path)
