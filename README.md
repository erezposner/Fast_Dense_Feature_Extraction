
# Fast Dense Feature Extraction for CNNs
An approach to compute patch-based local feature descriptors efficiently in
presence of pooling and striding layers for whole images at once.

A Pytorch implemention of the paper *"Fast Dense Feature Extraction with CNNs with Pooling Layers"*
https://arxiv.org/abs/1805.03096 

####Abstract
In recent years, many publications showed that convolutional neural network based features can have a superior performance to engineered features. However, not much effort was taken so far to extract local features efficiently for a whole image. In this paper, we present an approach to compute patch-based local feature descriptors efficiently in presence of pooling and striding layers for whole images at once. Our approach is generic and can be applied to nearly all existing network architectures. This includes networks for all local feature extraction tasks like camera calibration, Patchmatching, optical flow estimation and stereo matching. In addition, our approach can be applied to other patch-based approaches like sliding window object detection and recognition. We complete our paper with a speed benchmark of popular CNN based feature extraction approaches applied on a whole image, with and without our speedup, and example code (for Torch) that shows how an arbitrary CNN architecture can be easily converted by our approach.

![Arnon is the man](Paper/fast_dense.gif)

## Getting Started
These instructions will explain how to use the Fast Dense Feature Extraction (**FDFE**) project.

### Prerequisites
* Python>=3.5
* pytorch>=1.0
* tensorflow=2.0
* numpy
* matplotlib

### Installing

1) Install all prerequisites.
2) Clone the project

## Project Structure

* pytorch
    * ```FDFE.py``` - implementation of the all approach layers and pre & post process methods as described in the paper
    ,  including: 

        * MultiMaxPooling
        * MultiConv
        * multiPoolPrepare
        * unwarpPrepare
        * unwarpPool 
    * ```BaseNet.py``` - This referes to an implementation of a pre-trained CNN <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> on training patches <img src="https://latex.codecogs.com/svg.latex?\Small&space;P^{T}" />. 
    * ```SlimNet.py``` - This referes to the implementation of <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{I}" />.
    * ```sample_code.py``` - test run

* tf
    * ```FDFE.py``` - implementation of the all approach layers and pre & post process methods as described in the paper
    ,  including: 

        * MultiMaxPooling
        * MultiConv
        * multiPoolPrepare
        * unwarpPrepare
        * unwarpPool 
    * ```BaseNet.py``` - This referes to an implementation of a pre-trained CNN <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> on training patches <img src="https://latex.codecogs.com/svg.latex?\Small&space;P^{T}" />. 
    * ```SlimNet.py``` - This referes to the implementation of <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{I}" />.
    * ```sample_code.py``` - test run
    * tests
        * ```tf_tests.py``` - unit tests to check the output shapes of the FDFE layers
    
## Running the sample code


Now you should ```sample_code.py``` to make sure that FDFE project works correctly.

The test generates a random input image
<img src="https://latex.codecogs.com/svg.latex?\Small&space;I" /> 
 of size ```imH X imW```  and evaluates it on both 
<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> and <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{I}" />.

The script continues and evaluates differences between both CNN's outputs and performs speed benchmarking.

There are two modes of operation for 
<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" />:
* ***singlePatch*** mode- run <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> 
over a single patch ```pH x pW```
that would get cropped from input image <img src="https://latex.codecogs.com/svg.latex?\Small&space;I" /> around <img src="https://latex.codecogs.com/svg.latex?\Small&space;I(patch \space\space center_i,patch \space\space center_j)" /> 

* <b>allPatches</b> mode - run <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" />  over multuple patches at ones. 
here ```batch_size``` will determine how many patches would get evaluated at once.

 
### Possible arguments
In ```test_example.py``` there are initial parameters that could be adjusted: 
* Tested Input Image dimensions:
<pre>
- <b>imH</b> - Input image height  
- <b>imW</b> - Input image width

- <b>pW</b> - patch Width - current implementation supports only odd width size
- <b>pH</b> - patch Height - current implementation supports only odd width size

- <b>sL1</b> - First stride value 
- <b>sL2</b> - Second stride value
.
.
.
- <b>sLn</b> - n-th stride value
</pre>
 

* <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> <b>singlePatch</b> mode:

<pre>
- <b>patch_i_center</b> - patch row center
- <b>patch_j_center</b> - patch column center
</pre>
 
 * <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}"/> <b>allPatches</b> mode:
<pre>
- <b>batch_size</b> - number of patches to be evaluated at the same time 
</pre>
 


### Expected output
Script outputs the following:
* aggregated difference between base_net (<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" />) output and slim_net output
(<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{I}" />)
* For <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" />, an averaged evaluation per patch
* For <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{I}" />, Total evaluation per frame. i.e. the entire input image

Expected verbose would look like: (depends on <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> running  mode): 
<pre>
Total time for C_P: 0.017114248275756836 sec
------------------------------------------------------------
Averaged time for C_I per Patch without warm up: 0.0010887398617342114 sec
------- Comparison between a base_net over all patches output and slim_net -------
aggregated difference percentage = 0.0000000000 %
maximal abs difference = 0.0000000000 at index i=0,j=0
------------------------------------------------------------
</pre>
## To use your own network

In order to use your own pre-trained network that operates on patches you would need to:
* implemented your network in ```BaseNet.net```
* modify ```SlimNet.py``` accordingly:
    * Duplicate `BsetNet.py` model layers according to its order, e.g. 
    <pre>
    self.conv1 = list(base_net.modules())[<b>change_this_index</b>]</pre> 
    * For every `MaxPool2d` layer place `multiMaxPooling` instead with the decided stride value (<i>sLn</i>)
    * Deplicate unwrapPool layers according to the number of `multiMaxPooling` in your model
    *  Do not remove the following layers - multiPoolPrepare, unwrapPrepare
    
## Contributing

Contributions are always welcome! Please read the [contribution guidelines](contributing.md) first.
## Authors

* Erez P.  (erezposner@gmail.com)
* Arnon K. (arnon.kahani@gmail.com)


## Acknowledgments

A big thanks to the following individuals for designing the approach:

- Christian Bailer (christian.bailer@dfki.de)
- Tewodros A. Habtegebrial (tewodros_amberbir.habtegebrial@dfki.de)
- Kiran Varanasi1 (kiran.varanasi@dfki.de)
- Didier Stricker (didier.stricker@dfki.de)
