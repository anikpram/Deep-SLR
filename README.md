# Deep-SLR : Deep Generalization of Structured Low-Rank Algorithms
Tensorflow implementation of hybrid Deep-SLR (H-DSLR) for undersampled multi-channel MRI reconstruction. H-DSLR is a model-based deep learning approach to significantly
reduce the computational complexity of Structured low-rank (SLR) algorithms.
<img src="brain_6x_recon.png"  title="hover text">
# Code Details
The codes have been written in python-3.7 using the Tensorflow-1.15 platform. 

The description of the python scripts are:
1. trn_HDSLR.py : It is the training code. The number of iterations (K) of the data consistency and Dw block is set here.
2. tst_HDSLR.py : Code for testing the trained model on test dataset.
3. auxiliaryFunctions.py : Training and testing dataset preparation functions defined in this script.
4. HDSLR.py : Defines the H-DSLR network architecture to be trained.

