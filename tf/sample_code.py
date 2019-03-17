from time import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf.SlimNet import SlimNet
from tf.BaseNet import BaseNet


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def header_printer(text):
    print(bcolors.HEADER + text + bcolors.ENDC)


def normal_printer(text):
    print(text)


def bold_printer(text):
    print(bcolors.BOLD + text + bcolors.ENDC)


if __name__ == '__main__':
    device = 'gpu' if tf.test.is_gpu_available() else 'cpu'
    with tf.device("{}:0".format(device)):
        #######################################################
        # region debug mode
        methods = ['singlePatch', 'allPatches']
        running_mode = methods[1]  # options are ['singlePatch','allPatches']
        # for singlePatch mode define patch offset
        patch_j_center = 20
        patch_i_center = 42

        # for all patches define batch size
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

        # testImage = tf.random.uniform(shape=[1, imH, imW, 1],dtype=np.float)
        testImage = tf.zeros([1, imH, imW, 1])

        ## Adjust image dimensions to divide by S x S without remainder
        imW = int(np.ceil(imW / (sL1 * sL2)) * sL1 * sL2)
        imH = int(np.ceil(imH / (sL1 * sL2)) * sL1 * sL2)

        ## Base_net definitions
        base_net = BaseNet()
        base_net.build(input_shape=(None, pW, pH, 1,))
        ## SlimNet definitions & test run
        slim_net = SlimNet(base_net=base_net, pH=pH, pW=pW, sL1=sL1, sL2=sL2, imH=imH, imW=imW)

        # endregion
        #######################################################

        ## Perform slim_net evaluation on input image
        warm_up = 20
        iterations = 50
        times = []
        for i in range(0, warm_up + iterations + 1):
            start = time()
            slim_net_output = slim_net.call(testImage)
            times.append(time() - start)
        print('Total time for C_P: {}sec'.format(np.mean(times[warm_up:-1])))
        slim_net_output_numpy = slim_net_output

        ## Evaluate base_net over singlePatch / allPatches
        if (running_mode == methods[0]):  # evaluate base_net on a single patch
            testImage_np = np.squeeze(testImage.numpy(), axis=0)
            testPatch = testImage_np[patch_i_center - pH // 2:patch_i_center + pH // 2 + 1,
                        patch_j_center - pW // 2:patch_j_center + pW // 2 + 1]
            testPatch = testPatch
            testPatch = np.expand_dims(testPatch, axis=0)
            t = time()
            base_net_output = base_net.call(testPatch)
            p = time() - t
            base_net_output_numpy = base_net_output.numpy()
            header_printer('Total time for C_I per Patch without warm up: {}sec'.format(p))

            header_printer('------------------------------------------------------------')
            header_printer('------- Comparison between a base_net single patch evaluation and slim_net -------')
            bold_printer(
                'testPatch cropped at i={} j={} base_net output value is {}'.format(patch_i_center, patch_j_center,
                                                                                    base_net_output_numpy))
            bold_printer(
                'testPatch slim_net output value is {}'.format(
                    slim_net_output_numpy[..., patch_i_center, patch_j_center]))
            bold_printer('difference between the base_net & slim_net - {0:.13f}'.format(
                abs(slim_net_output_numpy[..., patch_i_center, patch_j_center] - base_net_output_numpy).sum()))

            header_printer('------------------------------------------------------------')
        if (running_mode == methods[1]):  # evaluate base_net over all patches

            # first pad input image in order to crop patches lying at the image boundary
            pady = pH - 1
            padx = pW - 1
            pad_top = np.ceil(pady / 2).astype(int)
            pad_bottom = np.floor(pady / 2).astype(int)
            pad_left = np.ceil(padx / 2).astype(int)
            pad_right = np.floor(padx / 2).astype(int)
            padded_testImage = tf.pad(testImage, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                                      "CONSTANT")

            # crop patches around all input image pixels
            x = np.arange(0, imW) + pad_left
            y = np.arange(0, imH) + pad_top
            X, Y = np.meshgrid(x, y)
            cropped_patches = [
                np.expand_dims(padded_testImage[0, y - pH // 2:y + pH // 2 + 1, x - pW // 2:x + pW // 2 + 1],axis=0)
                for x, y in zip(X.flatten(), Y.flatten())]
            time_array = []
            # evaluate base_net on all patches and build output image for comparison
            base_net_output_per_patch = np.zeros((slim_net.outChans, imH, imW))
            if batch_size > 1:
                iterations = int(np.ceil(len(cropped_patches) / batch_size))
                for p in range(0, iterations):
                    cropped_batches = tf.concat(cropped_patches[p * batch_size:(p + 1) * batch_size],axis=0)
                    t = time()
                    y_base = base_net.call(cropped_batches)
                    time_array.append(time() - t)
                    indices_in_batch = range(p * batch_size, (p + 1) * batch_size)[0:len(cropped_batches)]

                    yi, xi = np.unravel_index(indices_in_batch, (imH, imW))

                    base_current_output = np.squeeze(y_base.numpy(), axis=0)
                    for k in range(0, len(indices_in_batch)):
                        base_net_output_per_patch[..., yi[k], xi[k]] = base_current_output[k, ...]
            else:
                for ind, p in enumerate(cropped_patches):
                    p = p.to(device)
                    t = time()
                    y_base, = base_net.call(p)
                    time_array.append(time() - t)
                    yi, xi = np.unravel_index(ind, (imH, imW))
                    base_current_output = np.squeeze(y_base.numpy(), axis=0)
                    base_net_output_per_patch[..., yi, xi] = base_current_output

            header_printer('------------------------------------------------------------')
            header_printer('Averaged time for C_I per Patch without warm up: {}sec'.format(np.mean(time_array)))

            header_printer('------- Comparison between a base_net over all patches output and slim_net -------')
            bold_printer('aggregated difference percentage = {0:.10f}%'.format(
                (100 * np.sum(np.sum(abs(base_net_output_per_patch - slim_net_output_numpy)))) / (imH * imW)))
            index = np.argmax(abs(base_net_output_per_patch - slim_net_output_numpy))
            max_diff = np.max(abs(base_net_output_per_patch - slim_net_output_numpy))

            yi, xi = np.unravel_index(index, (imH, imW))

            header_printer('maximal abs difference = {0:.10f} at index i={1},j={2}'.format(max_diff, yi, xi))
            header_printer('------------------------------------------------------------')

            plt.close('all')
            plt.figure()
            plt.imshow((slim_net_output_numpy[0, ...].astype(float)))
            plt.title('slim')

            plt.figure()
            plt.imshow(base_net_output_per_patch[0, ...])
            plt.title('patch')

            plt.show()
