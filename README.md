# Deep-SLR : Deep Generalization of Structured Low-Rank Algorithms
Tensorflow implementation of hybrid Deep-SLR (H-DSLR) for undersampled multi-channel MRI reconstruction. H-DSLR is a model-based deep learning approach to significantly
reduce the computational complexity of Structured low-rank (SLR) algorithms.
<img src="brain_6x_recon.png"  title="hover text">
## Code Details
The codes have been written in python-3.7 using the Tensorflow-1.15 platform. 
### Training datasets
The training dataset consists of 12-channel brain MRI images from 4 subjects collected through SSFP acquisition protocol. There are 90 slices per subject which makes it 90 x 4 = 360 slices in total. It can be downloaded from the link https://drive.google.com/file/d/1Fml2PtQuECfbXAI86OYqzBb7K_CiQ2tk/view?usp=sharing . 

### Testing dataset
The testing dataset is uploaded as tst_img.npy which consists of a slice from another subject unseen by the network.

Description of the python scripts are:
1. trn_HDSLR.py : It is the training code. The number of iterations (K) of the data consistency and Dw block is set here.
2. tst_HDSLR.py : Code for testing the trained model on test dataset.
3. auxiliaryFunctions.py : Training and testing dataset preparation functions defined in this script.
4. HDSLR.py : Defines the H-DSLR network architecture to be trained.

