from tensorflow.keras import optimizers, callbacks, utils
from nets.SegNet import *
from PrepareData import prepare_data
import tensorflow as tf
import tensorflow_addons as tfa
import os


def train_by_fit(model, epochs, train_gen, test_gen, train_steps, test_steps):
    """
    fit方式训练
    :param model: 训练模型
    :param epochs: 训练轮数
    :param train_gen: 训练集生成器
    :param test_gen: 测试集生成器
    :param train_steps: 训练次数
    :param test_steps: 测试次数
    :return: None
    """

    cbk = [
        callbacks.ModelCheckpoint(
            './backend/best_weights.h5',
            save_weights_only=True, save_best_only=True)
    ]

    optimizer = tfa.optimizers.RectifiedAdam(lr=1e-3)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # trainable_layer = 92
    trainable_layer = 19
    for i in range(trainable_layer):
        print(model.layers[i].name)
        model.layers[i].trainable = False
    print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=test_gen,
              validation_steps=test_steps,
              epochs=epochs,
              callbacks=cbk,
              shuffle=True)

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    print('train all layers.')

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=test_gen,
              validation_steps=test_steps,
              epochs=epochs * 2,
              initial_epoch=epochs,
              callbacks=cbk,
              shuffle=True)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    model = SegNet((128, 128, 3), 151)
    model.load_weights('./backend/best_weights.h5')
    model.summary()

    dataset, TRAINSET_SIZE, VALSET_SIZE, BATCH_SIZE = prepare_data("./data/ADEChallengeData2016/images/")
    train_steps = TRAINSET_SIZE // BATCH_SIZE
    test_steps = VALSET_SIZE // BATCH_SIZE
    
    train_gen = dataset['train']
    test_gen = dataset['val']

    train_by_fit(model, 40, train_gen, test_gen, train_steps, test_steps)
