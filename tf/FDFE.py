import tensorflow as tf

# tf.enable_eager_execution()

import numpy as np


# (B,H,W,C)

class multiPoolPrepare(tf.keras.layers.Layer):
    def __init__(self, patchY, patchX):
        super(multiPoolPrepare, self).__init__()

        pady = patchY - 1
        padx = patchX - 1

        _pad_top = np.ceil(pady / 2).astype(int)
        _pad_bottom = np.floor(pady / 2).astype(int)
        _pad_left = np.ceil(padx / 2).astype(int)
        _pad_right = np.floor(padx / 2).astype(int)
        self.padding = tf.constant([[0, 0], [_pad_top, _pad_bottom], [_pad_left, _pad_right], [0, 0]], name='padding')

    def call(self, input):
        # Override call() instead of __call__ so we can perform some bookkeeping.
        return tf.pad(input, self.padding, "CONSTANT")


class unwrapPrepare(tf.keras.layers.Layer):
    def __init__(self):
        super(unwrapPrepare, self).__init__()
        self.crop = tf.constant([0, 0, 0, 0], name='unwrapPrepare_crop')

    def call(self, input):
        x = tf.slice(input, self.crop, [input.shape[0], input.shape[1] - 1, input.shape[2] - 1, input.shape[3]])
        x = tf.reshape(x, [x.shape[0], -1], name='unwrapPrepare_reshape')
        # y = x
        y = tf.transpose(x)
        return y


class unwrapPool(tf.keras.layers.Layer):
    def __init__(self, outChans, curImgW, curImgH, dW, dH):
        super(unwrapPool, self).__init__()
        self.outChans = int(outChans)
        self.curImgW = int(curImgW)
        self.curImgH = int(curImgH)
        self.dW = int(dW)
        self.dH = int(dH)

    def call(self, input):
        x = tf.reshape(input, [self.outChans, self.curImgW, self.curImgH, self.dH, self.dW, -1])
        y = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
        return y


class multiMaxPooling(tf.keras.layers.Layer):
    def __init__(self, kW, kH, dW, dH):
        super(multiMaxPooling, self).__init__()
        layers = []
        self.padd = []
        for i in range(0, dH):
            for j in range(0, dW):
                self.padd.append((-j, -i))
                layers.append(tf.keras.layers.MaxPool2D(pool_size=(kW, kH), strides=(dW, dH)))
        self.max_layers = layers  # nn.ModuleList(layers)
        self.s = dH

    def call(self, input):

        hh = []
        ww = []
        res = []

        for i in range(0, len(self.max_layers)):
            pad_left, pad_top = self.padd[i]
            b, h, w, c = input.shape.as_list()

            _x = tf.slice(input, begin=[0, pad_top * -1, pad_left * -1, 0], size=[b, h + pad_top, w + pad_left, c])
            # _x = F.pad(input, [pad_left, pad_left, pad_top, pad_top], value=0)
            _x = self.max_layers[i](_x)
            _, h, w, _ = _x.shape.as_list()
            hh.append(h)
            ww.append(w)
            res.append(_x)
        max_h, max_w = np.max(hh), np.max(ww)
        for i in range(0, len(self.max_layers)):
            _x = res[i]
            _, h, w, _ = _x.shape.as_list()
            pad_top = np.floor((max_h - h) / 2).astype(int)
            pad_bottom = np.ceil((max_h - h) / 2).astype(int)
            pad_left = np.floor((max_w - w) / 2).astype(int)
            pad_right = np.ceil((max_w - w) / 2).astype(int)
            _x = tf.pad(_x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
            res[i] = _x
        return tf.concat(res, 0)


class multiConv(tf.keras.layers.Layer):
    def __init__(self, nOutputPlane, kW, kH, dW, dH):
        super(multiConv, self).__init__()
        layers = []
        self.padd = []
        for i in range(0, dH):
            for j in range(0, dW):
                self.padd.append((-j, -i))
                layers.append(tf.keras.layers.Conv2D(filters=nOutputPlane, kernal_size=(kW, kH), strides=(dW, dH)))
        self.max_layers = layers  # nn.ModuleList(layers)
        self.s = dH

    def call(self, input):

        hh = []
        ww = []
        res = []

        for i in range(0, len(self.max_layers)):
            pad_left, pad_top = self.padd[i]
            b, h, w, c = input.shape.as_list()

            _x = tf.slice(input, begin=[0, pad_top * -1, pad_left * -1, 0], size=[b, h + pad_top, w + pad_left, c])
            # _x = F.pad(input, [pad_left, pad_left, pad_top, pad_top], value=0)
            _x = self.max_layers[i](_x)
            _, h, w, _ = _x.shape.as_list()
            hh.append(h)
            ww.append(w)
            res.append(_x)
        max_h, max_w = np.max(hh), np.max(ww)
        for i in range(0, len(self.max_layers)):
            _x = res[i]
            _, h, w, _ = _x.shape.as_list()
            pad_top = np.floor((max_h - h) / 2).astype(int)
            pad_bottom = np.ceil((max_h - h) / 2).astype(int)
            pad_left = np.floor((max_w - w) / 2).astype(int)
            pad_right = np.ceil((max_w - w) / 2).astype(int)
            _x = tf.pad(_x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
            res[i] = _x
        return tf.concat(res, 0)


### Testing ###


# imH = 960
# imW = 1280
#
#
# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
#
#
# def unit_test_layer(layer, input, output):
#     print(bcolors.HEADER + "Testing layer: {}".format(layer.name) + bcolors.ENDC)
#     res = layer(input).shape
#     if (res == output.shape):
#         print(bcolors.OKGREEN + "Pass" + bcolors.ENDC)
#     else:
#         print(bcolors.FAIL + "Fail -  expected: {} got: {}".format(output.shape, res) + bcolors.ENDC)




