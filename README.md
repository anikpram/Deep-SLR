# Deep-SLR : Deep Generalization of Structured Low-Rank Algorithms
Tensorflow implementation of hybrid Deep-SLR (H-DSLR) for undersampled multi-channel MRI reconstruction. H-DSLR is a model-based deep learning approach to significantly
reduce the computational complexity of Structured low-rank (SLR) algorithms.
<img src="brain_6x_recon.png"  title="hover text">
## Code Details
The codes have been written in python-3.7 using the Tensorflow-1.15 platform. 
### Training datasets
The training dataset consists of 12-channel brain MRI images from 4 subjects collected through SSFP acquisition protocol. There are 90 slices per subject which makes it 90 x 4 = 360 slices in total. Each slice is of dimension 256 x 232 x 12. The dataset can be downloaded from the link https://drive.google.com/file/d/1Fml2PtQuECfbXAI86OYqzBb7K_CiQ2tk/view?usp=sharing . 

### Testing dataset
The testing dataset is uploaded as tst_img.npy which consists of a slice from another subject unseen by the network.

Description of the python scripts are:
1. trn_HDSLR.py : It is the training code. The H-DSLR model is trained on the 360 12-channel brain slices described above.
2. tst_HDSLR.py : It is the code for testing a pre-trained model on the test dataset uploaded as tst_img.npy. The pre-trained model is inside the directory 'savedModels'.
3. auxiliaryFunctions.py : The training and testing dataset preparation related functions are defined in this script.
4. HDSLR.py : Defines the H-DSLR network architecture to be trained.

## Reference
Pramanik, Aniket, Hemant Aggarwal, and Mathews Jacob. "Deep Generalization of Structured Low-Rank Algorithms (Deep-SLR).", arXiv preprint arXiv:1912.03433 (2020).\\
Arxiv Link: https://arxiv.org/pdf/1912.03433.pdf
