![Arnon is the man](Paper/fast_dense.gif)
# Fast Dense Feature Extraction for CNNs

A Pytorch implemention of the paper *"Fast Dense Feature Extraction with CNNs with Pooling Layers"*
https://arxiv.org/abs/1805.03096 

An approach to compute patch-based local feature descriptors efficiently in
presence of pooling and striding layers for whole images at once.
## Getting Started

These instructions will explain how to use the Fast Dense Feature Extraction (**FDFE**) project.

### Prerequisites
* Python>=3.5
* pytorch>=1.0
* numpy
* matplotlib

### Installing

1) Install all prerequisites.
2) Clone the project

## Project Structure


* ```FDFE.py``` - implementation of the all approach layers and pre & post process methods as described in the paper
,  including: 

  * MultiMaxPooling
  * MultiConv
  * multiPoolPrepare
  * unwarpPrepare
  * unwarpPool 
* ```BaseNet.py``` - This referes to an implementation of a pre-trained CNN 
<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> 
on training patches 
<img src="https://latex.codecogs.com/svg.latex?\Small&space;P^{T}" />
. 
* ```SlimNet.py``` - This referes to the implementation of
  <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{I}" />
.
* ```test_example.py``` - test run


## Running the tests


Now you should ```test_example.py``` to make sure that FDFE project works correctly.

The test generates a random input image
<img src="https://latex.codecogs.com/svg.latex?\Small&space;I" /> 
 of size ```imH X imW```  and evaluates it on both 
<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" />  
<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{I}" />
.

The script continues and evaluates differences between both CNN's outputs and performs speed benchmarking.

There are two modes of operation for 
<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> 
:
* ***singlePatch*** mode- run 
<img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> 
over a single patch ```pH x pW```
that would get cropped from input image
<img src="https://latex.codecogs.com/svg.latex?\Small&space;I" /> 
around ```I(patch_j_center,patch_i_center)```

* allPatches

 
### Possible arguments
In ```test_example.py``` there are initial parameters that could be adjusted: 
* Tested Input Image dimensions:
<pre>
- <b>imH</b> - Input image height  
- <b>imW</b> - Input image width
</pre>
 

* <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}" /> <b>singlePatch</b> mode:

<pre>
- <b>pW</b> - patch Width - current implementation supports only odd width size
- <b>pH</b> - patch Height - current implementation supports only odd width size
- <b>patch_i_center</b> - patch row center
- <b>patch_j_center</b> - patch column center
</pre>
 
 * <img src="https://latex.codecogs.com/svg.latex?\Small&space;C_{p}"/> <b>allPatches</b> mode:
<pre>
- <b>batch_size</b> - number of patches to be evaluated at the same time 
</pre>
 


### Expected output

```

```
## To use your own network

adadadkbdabkjbdfajksbd

Add additional notes about how to deploy this on a live system

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Erez P.** - *Initial work*
* **Arnon K.** - *Initial work*


## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
