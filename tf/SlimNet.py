import tensorflow as tf

from tf.BaseNet import BaseNet
from tf.FDFE import multiPoolPrepare, multiMaxPooling, unwrapPrepare, unwrapPool

import numpy as np


class SlimNet(tf.keras.Model):
    def __init__(self, base_net, pH, pW, sL1, sL2, imH, imW):
        super(SlimNet, self).__init__()
        self.imH = imH
        self.imW = imW
        self.multiPoolPrepare = multiPoolPrepare(pH, pW)
        self.conv1 = base_net.layers[0]
        self.multiMaxPooling1 = multiMaxPooling(sL1, sL1, sL1, sL1)
        self.conv2 = base_net.layers[2]
        self.multiMaxPooling2 = multiMaxPooling(sL2, sL2, sL2, sL2)
        self.conv3 = base_net.layers[4]
        self.conv4 = base_net.layers[5]
        self.outChans = 113
        self.unwrapPrepare = unwrapPrepare()
        self.unwrapPool2 = unwrapPool(self.outChans, imH / (sL1 * sL2), imW / (sL1 * sL2), sL2, sL2)
        self.unwrapPool3 = unwrapPool(self.outChans, imH / sL1, imW / sL1, sL1, sL1)
        self.reshape_end = tf.keras.layers.Reshape((-1, self.imH, self.imW),name="output_123")

    def call(self, inputs):
        x = self.multiPoolPrepare(inputs)

        x = self.conv1(x)

        x = self.multiMaxPooling1(x)

        x = self.conv2(x)

        x = self.multiMaxPooling2(x)
        x = self.conv3(x)

        x = self.conv4(x)
        x = self.unwrapPrepare(x)
        x = self.unwrapPool2(x)
        x = self.unwrapPool3(x)
        y = self.reshape_end(x)

        return y


if __name__ == '__main__':
    print(tf.test.is_gpu_available())
    with tf.device("/gpu:0"):
        batch_size = 10

        # endregion
        #######################################################
        # region Initial Parameters

        ## Image Dimensions
        imH = 960
        imW = 1280

        ## Stride step size
        sL1 = 2
        sL2 = 2

        ## define patch dimensions
        pW = 15
        pH = 15

        # endregion
        #######################################################
        # region setup ground

        testImage = tf.zeros([1, imH, imW, 1])

        ## Adjust image dimensions to divide by S x S without remainder
        imW = int(np.ceil(imW / (sL1 * sL2)) * sL1 * sL2)
        imH = int(np.ceil(imH / (sL1 * sL2)) * sL1 * sL2)

        ## Base_net definitions
        base_net = BaseNet()
        base_net.build((1, imH, imW, 1))
        ## SlimNet definitions & test run
        slim_net = SlimNet(base_net=base_net, pH=pH, pW=pW, sL1=sL1, sL2=sL2, imH=imH, imW=imW)

        SlimNet_test_input = tf.zeros([1, imH, imW, 1])
        from time import time
        values = []
        for i in range(10):
            start = time()
            _output = slim_net(SlimNet_test_input)
            _output = tf.transpose(_output, perm=(1, 0, 2, 3))
            values.append(time() - start)
        print(np.average(values[10:]))

        print(_output.shape)

        checkpoint = tf.train.Checkpoint(x=slim_net)
        checkpoint.save('./ckpt/')

