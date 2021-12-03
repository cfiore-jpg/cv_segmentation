'''
Description:
    Prepare the data for training and testing.
'''
import numpy as np
import skimage.io
import scipy.io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import os

# ==========================================================
#                      !!! WARNING !!!
#
# For now, I just take images from 2008 and
# corresponding annotations for model training.
# You may want to change the code when you want more data.
# ==========================================================


def prepareData(img_filename, annotation_filename):
    # load data
    img_data = np.load(img_filename, allow_pickle=True)
    annot_data = np.load(annotation_filename, allow_pickle=True)

    # TODO: for now, take just the first 100 images as an eaxmple
    img_data = img_data[:50]
    annot_data = annot_data[:50]

    # preprocessing
    img_data = img_data / 255
    annot_data = np.asarray(annot_data, dtype=np.int32)

    # Split into train and test set
    train_idx, test_idx = train_test_split(np.arange(img_data.shape[0]), train_size=0.7)
    train_X = img_data[train_idx]
    train_Y = annot_data[train_idx]
    test_X = img_data[test_idx]
    test_Y = annot_data[test_idx]
    return train_X, test_X, train_Y, test_Y



if __name__ == '__main__':
    img_filename = "./selected_img_data.npy"
    annotation_filename = "./selected_annot_data.npy"
    train_X, test_X, train_Y, test_Y = prepareData(img_filename, annotation_filename)