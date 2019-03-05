from time import time
import torch
import numpy as np
import torch.nn.functional as F

from pytorch.SlimNet import SlimNet
from pytorch.BaseNet import BaseNet

if __name__ == '__main__':
    #######################################################
    # region debug mode
    methods = ['singlePatch', 'allPatches']
    running_mode = methods[0]  # options are ['singlePatch','allPatches']
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
    device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")

    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    testImage = torch.randn(1, 1, imH, imW)
    testImage = testImage.cuda().half()

    ## Adjust image dimensions to divide by S x S without remainder
    imW = int(np.ceil(imW / (sL1 * sL2)) * sL1 * sL2)
    imH = int(np.ceil(imH / (sL1 * sL2)) * sL1 * sL2)

    ## Base_net definitions
    base_net = BaseNet()
    base_net.cuda()
    base_net.eval()
    ## SlimNet definitions & test run
    slim_net = SlimNet(base_net=base_net, pH=pH, pW=pW, sL1=sL1, sL2=sL2, imH=imH, imW=imW)
    slim_net.cuda()
    slim_net.eval()
    slim_net.half()
    # endregion
    #######################################################

    ## Perform slim_net evaluation on input image
    warm_up = 10
    iterations = 50
    times = []
    for i in range(0, warm_up + iterations + 1):
        start = time()
        slim_net_output = slim_net(testImage)
        times.append(time() - start)
    print('Total time for C_P: {}sec'.format(np.mean(times[warm_up:-1])))
    slim_net_output_numpy = slim_net_output.detach().cpu().numpy()

    ## Evaluate base_net over singlePatch / allPatches
    if (running_mode == methods[0]):  # evaluate base_net on a single patch
        testImage_np = testImage.squeeze().cpu().numpy()
        testPatch = testImage_np[patch_i_center - pH // 2:patch_i_center + pH // 2 + 1,
                    patch_j_center - pW // 2:patch_j_center + pW // 2 + 1]
        testPatch = torch.from_numpy(testPatch)
        testPatch = testPatch.unsqueeze(0)
        testPatch = testPatch.unsqueeze(0)
        testPatch = testPatch.to(device)
        t = time()
        base_net_output = base_net(testPatch)
        p = time() - t
        base_net_output_numpy = base_net_output.detach().cpu().numpy().squeeze()
        print('Total time for C_I per Patch without warm up: {}sec'.format(p))

        print('------------------------------------------------------------')
        print('------- Comparison between a base_net single patch evaluation and slim_net -------')
        print('testPatch cropped at i={} j={} base_net output value is {}'.format(patch_i_center, patch_j_center,
                                                                                  base_net_output_numpy))
        print(
            'testPatch slim_net output value is {}'.format(slim_net_output_numpy[..., patch_i_center, patch_j_center]))
        print('difference between the base_net & slim_net - {0:.13f}'.format(
            abs(slim_net_output_numpy[..., patch_i_center, patch_j_center] - base_net_output_numpy).sum()))

        print('------------------------------------------------------------')
    if (running_mode == methods[1]):  # evaluate base_net over all patches

        # first pad input image in order to crop patches lying at the image boundary
        pady = pH - 1
        padx = pW - 1
        pad_top = np.ceil(pady / 2).astype(int)
        pad_bottom = np.floor(pady / 2).astype(int)
        pad_left = np.ceil(padx / 2).astype(int)
        pad_right = np.floor(padx / 2).astype(int)
        padded_testImage = F.pad(testImage, [pad_left, pad_right, pad_top, pad_bottom], value=0)

        # crop patches around all input image pixels
        x = np.arange(0, imW) + pad_left
        y = np.arange(0, imH) + pad_top
        X, Y = np.meshgrid(x, y)
        cropped_patches = [
            padded_testImage[0, 0, y - pH // 2:y + pH // 2 + 1, x - pW // 2:x + pW // 2 + 1].unsqueeze(0).unsqueeze(0)
            for x, y in zip(X.flatten(), Y.flatten())]
        time_array = []
        # evaluate base_net on all patches and build output image for comparison
        base_net_output_per_patch = np.zeros((slim_net.outChans, imH, imW))
        if batch_size > 1:
            iterations = int(np.ceil(len(cropped_patches) / batch_size))
            for p in range(0, iterations):
                cropped_batches = torch.cat(cropped_patches[p * batch_size:(p + 1) * batch_size])
                cropped_batches = cropped_batches.to(device)
                t = time()
                y_base = base_net(cropped_batches)
                time_array.append(time() - t)
                indices_in_batch = range(p * batch_size, (p + 1) * batch_size)[0:len(cropped_batches)]

                yi, xi = np.unravel_index(indices_in_batch, (imH, imW))

                base_current_output = y_base.detach().cpu().numpy().squeeze()
                for k in range(0, len(indices_in_batch)):
                    base_net_output_per_patch[..., yi[k], xi[k]] = base_current_output[k, ...]
        else:
            for ind, p in enumerate(cropped_patches):
                p = p.to(device)
                t = time()
                y_base, = base_net(p)
                time_array.append(time() - t)
                yi, xi = np.unravel_index(ind, (imH, imW))
                base_current_output = y_base.detach().cpu().numpy().squeeze()
                base_net_output_per_patch[..., yi, xi] = base_current_output

        print('------------------------------------------------------------')
        print('Averaged time for C_I per Patch without warm up: {}sec'.format(np.mean(time_array)))

        print('------- Comparison between a base_net over all patches output and slim_net -------')
        print('aggregated difference percentage = {0:.10f}%'.format(
            (100 * np.sum(np.sum(abs(base_net_output_per_patch - slim_net_output_numpy)))) / (imH * imW)))
        index = np.argmax(abs(base_net_output_per_patch - slim_net_output_numpy))
        max_diff = np.max(abs(base_net_output_per_patch - slim_net_output_numpy))

        yi, xi = np.unravel_index(index, (imH, imW))

        print('maximal abs difference = {0:.10f} at index i={1},j={2}'.format(max_diff, yi, xi))
        print('------------------------------------------------------------')

        import matplotlib.pyplot as plt

        plt.close('all')
        plt.figure()
        plt.imshow((slim_net_output_numpy[0, ...].astype(float)))
        plt.title('slim')

        plt.figure()
        plt.imshow(base_net_output_per_patch[0, ...])
        plt.title('patch')

        plt.show()
