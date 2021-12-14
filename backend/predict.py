# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2021/5/12 16:39
# @Software: PyCharm
# @Brief: 预测脚本
import backend.core.config as cfg
from backend.core.dataset import resize_image_with_pad
from backend.nets.SegNet import *
import tensorflow as tf
import os
from PIL import Image
import cv2 as cv
import numpy as np


def inference(model, image):
    """
    前向推理
    :param model: 模型对象
    :param image:  输入图像
    :return:
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = resize_image_with_pad(image, target_size=cfg.input_shape[:2])
    image = np.expand_dims(image, axis=0)
    image -= [123.68, 116.68, 103.94]

    pred_mask = model.predict(image)
    pred_mask = tf.nn.softmax(pred_mask)

    pred_mask = np.squeeze(pred_mask)
    pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)

    return pred_mask


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # model = SegNet(cfg.input_shape, cfg.num_classes)
    model = SegNet_VGG16(cfg.input_shape, cfg.num_classes)
    model.load_weights("segnet_weights.h5")

    image = cv.imread("/Users/marti/Desktop/ORange HAt.jpg")

    result = inference(model, image)

    result = Image.fromarray(result, mode='P')
