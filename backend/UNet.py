'''
Description:
    Define the UNet architecture.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, Cropping2D
from tensorflow.keras.optimizers import Adam


class UNet(tf.keras.Model):
    
    def __init__(self, filters_dims, num_label):
        super(UNet, self).__init__()
        # Encoding Phase
        self.encoding_conv_phase = []
        self.encoding_pool_phase = []
        for i in range(len(filters_dims) - 1):
            self.encoding_conv_phase.append([
                Conv2D(filters_dims[i], 3, activation="relu", padding="same", name="encoding_block{}_conv1".format(i+1)),
                Conv2D(filters_dims[i], 3, activation="relu", padding="same", name="encoding_block{}_conv2".format(i+1))
            ])
            self.encoding_pool_phase.append(MaxPooling2D(pool_size=(2, 2), padding="same", name="encoding_block{}_pool".format(i+1)))
        # Middle Phase
        self.middle_phase = []
        self.middle_phase.append(Conv2D(filters_dims[-1], 3, activation="relu", padding="same", name="middle_block_conv1"))
        self.middle_phase.append(Conv2D(filters_dims[-1], 3, activation="relu", padding="same", name="middle_block_conv2"))
        self.middle_phase.append(Dropout(0.5))
        # Decoding Phase
        self.decoding_conv_phase = []
        self.decoding_upsampling_phase = []
        filters_dims = filters_dims[::-1]
        for i in range(1, len(filters_dims)):
            self.decoding_upsampling_phase.append([
                UpSampling2D(size=(2, 2), name="decoding_up{}".format(i+1)),
                Conv2D(filters_dims[i], 3, activation="relu", padding="same", name="decoding_up{}_conv".format(i+1))
            ])
            self.decoding_conv_phase.append([
                Conv2D(filters_dims[i], 3, activation="relu", padding="same", name="decoding_block{}_conv1".format(i+1)),
                Conv2D(filters_dims[i], 3, activation="relu", padding="same", name="decoding_block{}_conv2".format(i+1))
            ])
        # Prediction output
        self.prediction_head = Conv2D(num_label, 1, activation='softmax', kernel_initializer='glorot_uniform')
        # --------------------------------------------------------------
        self.batch_size = 8
        self.optimizer = Adam(learning_rate=1e-4)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


    def call(self, X):
        output = X
        # encoding phase
        encoding_output = []
        for i in range(len(self.encoding_conv_phase)):
            for conv_layer in self.encoding_conv_phase[i]:
                output = conv_layer(output)
            encoding_output.append(output)
            output = self.encoding_pool_phase[i](output)
        # middle phase
        for i in range(len(self.middle_phase)):
            output = self.middle_phase[i](output)
        # decoding phase
        encoding_output = encoding_output[::-1]
        for i in range(len(self.decoding_conv_phase)):
            for up_layer in self.decoding_upsampling_phase[i]:
                output = up_layer(output)
            # output = self.cropping[i](output)
            output = self._crop(output, encoding_output[i]) #TODO: check this!
            output = tf.concat([encoding_output[i], output], axis=3)
            for conv_layer in self.decoding_conv_phase[i]:
                output = conv_layer(output)
        # prediction
        pred = self.prediction_head(output)
        return pred


    def loss(self, prob, true_label):
        return self.loss_function(true_label, prob)


    def accuracy(self, prob, true_label):
        pred_label = tf.argmax(prob, axis=3)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_label, true_label), tf.float32))
        return acc


    def _crop(self, source, target):
        target_shape = np.asarray(target.shape)[[1,2]]
        source_shape = np.asarray(source.shape)[[1,2]]
        dim1_diff = source_shape[0] - target_shape[0]
        dim2_diff = source_shape[1] - target_shape[1]
        return Cropping2D(((0, dim1_diff), (0, dim2_diff)))(source)

if __name__ == '__main__':
    from Running import train
    # Sanity check the model on randomly generated data
    input_shape = [375, 500, 3]
    num_labels = 5
    num_samples = 16
    imgs = np.random.uniform(0.0, 1.0, [num_samples] + input_shape)
    true_label = np.random.choice(np.arange(num_labels), [num_samples] + input_shape[:-1], replace=True)

    filters_dims = [64, 128, 256, 512]
    model = UNet(filters_dims, num_label=5)
    # pred = model.call(imgs)
    # loss = model.loss(pred, true_label)
    # acc = model.accuracy(pred, true_label)
    # print("Loss = {}".format(loss))
    # print("Acc = {}".format(acc))

    train(model, imgs, true_label)
