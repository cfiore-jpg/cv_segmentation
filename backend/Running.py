'''
Description:
    Train and evaluate the model.
'''
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.models import load_model
import tensorflow as tf
from PrepareData import prepareData
from UNet import UNet



def train(model, train_input, train_label, num_epochs):
    time_str = datetime.datetime.now().strftime("%I%M%p_%Y%B%d")
    batch_size = model.batch_size
    num_samples = len(train_label)
    epoch_loss_list = []
    epoch_accuracy_list = []
    for t in range(num_epochs):
        print("-" * 70)
        loss_list = []
        accuracy_list = []
        sum_loss = 0.0
        for bi, idx in enumerate(range(0, num_samples, batch_size)):
            batch_img = train_input[idx: min(idx + batch_size, num_samples)]
            batch_label = train_label[idx: min(idx + batch_size, num_samples)]
            with tf.GradientTape() as tape:
                pred_Y = model(batch_img)
                loss = model.loss(pred_Y, batch_label)
                acc = model.accuracy(pred_Y, batch_label)
                sum_loss += loss
            loss_list.append(loss)
            accuracy_list.append(acc)
            print('Batch %d\tLoss: %.3f | Acc: %.3f' % (bi, loss, acc))
            # Update the model after every batch
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("*" * 70)
        print('Epoch %d\tLoss: %.3f | Acc: %.3f' % (t, np.mean(loss_list), np.mean(accuracy_list)))
        epoch_loss_list.append(loss_list)
        epoch_accuracy_list.append(accuracy_list)
        # save model for every epoch
        checkpoint_dir = "{}".format(time_str)
        if checkpoint_dir not in os.listdir("./checkpoint"):
            os.mkdir("./checkpoint/{}".format(checkpoint_dir))
        model.save("./checkpoint/{}/UNet-epoch{}".format(checkpoint_dir, t))
    return model, epoch_loss_list, epoch_accuracy_list


def evaluate(model, test_input, test_label):
    '''
    This loss and accuracy function is the same as it in the UNet class.
    '''
    def loss(prob, true_label):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(true_label, prob)

    def accuracy(prob, true_label):
        pred_label = tf.argmax(prob, axis=3)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_label, true_label), tf.float32))
        return acc

    pred_Y = model(test_input)
    loss_val = loss(pred_Y, test_label)
    acc = accuracy(pred_Y, test_label)
    print('Testing \tLoss: %.3f | Acc: %.3f' % (loss_val, acc))


def main(img_filename, annotation_filename, num_epochs=5, need_load=False):
    # prepare data
    print("=" * 70)
    print("Preparing data...")
    train_X, test_X, train_Y, test_Y = prepareData(img_filename, annotation_filename)
    print("Number of training data : {}".format(len(train_Y)))
    print("Number of testing data : {}".format(len(test_Y)))
    # build the model
    label_size = 459
    filters_dims = [64, 128, 256, 512]
    model = UNet(filters_dims, label_size)
    # training
    if not need_load:
        print("=" * 70)
        print("Start training...")
        model, epoch_loss_list, epoch_accuracy_list = train(model, train_X, train_Y, num_epochs)
    else:
        model = load_model("./checkpoint/0509PM_2021December03/UNet-epoch0")
    # testing
    print("=" * 70)
    print("Start testing...")
    evaluate(model, test_X, test_Y)



if __name__ == '__main__':
    img_filename = "./selected_img_data.npy"
    annotation_filename = "./selected_annot_data.npy"
    main(img_filename, annotation_filename, num_epochs=5, need_load=True)