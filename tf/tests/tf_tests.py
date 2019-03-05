import unittest
import tensorflow as tf

from tf.FDFE import multiPoolPrepare, unwrapPrepare, unwrapPool, multiMaxPooling

tf.enable_eager_execution()

imH = 960
imW = 1280


class TestingTensorflowFDFE(unittest.TestCase):

    def test_multiMaxPooling(self):
        multiMaxPooling_test_input = tf.zeros([4, 484, 644, 16])
        multiMaxPooling_test_output = tf.zeros([16, 242, 322, 16])

        layer_multiMaxPooling = multiMaxPooling(2, 2, 2, 2)
        _output = layer_multiMaxPooling(multiMaxPooling_test_input)
        self.assertEqual(_output.shape, multiMaxPooling_test_output.shape)

    def test_unwrapPrepare(self):
        unwrapPrepare_test_input = tf.zeros([16, 241, 321, 113])
        unwrapPrepare_test_output = tf.zeros([8678400, 16])

        layer_unwrapPrepare = unwrapPrepare()
        _output = layer_unwrapPrepare(unwrapPrepare_test_input)
        self.assertEqual(_output.shape, unwrapPrepare_test_output.shape)

    def test_multiPoolPrepare(self):
        multiPoolPrepare_test_input = tf.zeros([1, imH, imW, 1])
        multiPoolPrepare_test_output = tf.zeros([1, 974, 1294, 1])
        layer_multiPoolPrepare = multiPoolPrepare(15, 15)
        _output = layer_multiPoolPrepare(multiPoolPrepare_test_input)
        self.assertEqual(_output.shape, multiPoolPrepare_test_output.shape)

    def test_unwrapPool(self):
        unwrapPool_test_input = tf.zeros([8678400, 16])
        unwrapPool_test_output = tf.zeros([113, 240, 2, 320, 2, 4])

        layer_unwrapPool = unwrapPool(113, imH / (2 * 2), imW / (2 * 2), 2, 2)
        _output = layer_unwrapPool(unwrapPool_test_input)
        self.assertEqual(_output.shape, unwrapPool_test_output.shape)

        unwrapPool_test_input = tf.zeros([8678400, 16])
        unwrapPool_test_output = tf.zeros([113, 240, 2, 320, 2, 4])

        layer_unwrapPool = unwrapPool(113, imH / (2 * 2), imW / (2 * 2), 2, 2)
        _output = layer_unwrapPool(unwrapPool_test_input)
        self.assertEqual(_output.shape, unwrapPool_test_output.shape)


if __name__ == '__main__':
    unittest.main()
