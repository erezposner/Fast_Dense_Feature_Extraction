import tensorflow as tf

# tf.enable_eager_execution()


class BaseNet(tf.keras.Model):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='relu')

        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=2)  # nn.MaxPool2d(kernel_size=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu')

        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=1, activation='relu')

        self.conv4 = tf.keras.layers.Conv2D(filters=113, kernel_size=1, strides=1)  # nn.Conv2d(128, 113, kernel_size=1)

    def call(self, inputs):
        x = self.conv1(inputs)

        x = self.max_pool1(x)
        x = self.conv2(x)

        x = self.max_pool2(x)
        x = self.conv3(x)

        y = self.conv4(x)
        return y

if __name__ == '__main__':
    BaseNet_test_input = tf.zeros([1, 15, 15, 1])
    model = BaseNet()
    model(BaseNet_test_input)
