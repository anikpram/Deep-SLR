# Deep-SLR : Deep Generalization of Structured Low-Rank Algorithms
Structured low-rank (SLR) algorithms, which exploit annihilation relations between the Fourier samples of a signal resulting from different properties, is a powerful image reconstruction framework in several applications. This scheme relies on low-rank matrix completion to estimate the annihilation relations from the measurements. The main challenge with this strategy is the high computational complexity of matrix completion. We introduce a deep learning (DL) approach to
significantly reduce the computational complexity. Specifically, we use a convolutional neural network (CNN)-based filterbank that is trained to estimate the annihilation relations from imperfect (under-sampled and noisy) k-space measurements of Magnetic Resonance Imaging (MRI). The main reason for the computational efficiency is the pre-learning of the parameters of the non-linear CNN from exemplar data, compared to SLR schemes that learn the linear filterbank parameters from the dataset itself. Experimental comparisons show that the proposed scheme can enable calibration-less parallel MRI; it can offer performance similar to SLR schemes while reducing the runtime by around three orders of magnitude. Unlike pre-calibrated and self-calibrated approaches, the proposed uncalibrated approach
is insensitive to motion errors and affords higher acceleration. The proposed scheme also incorporates image domain priors that are complementary, thus significantly improving the performance over that of SLR schemes.

## Implementation
Tensorflow implementation of hybrid Deep-SLR (H-DSLR) network for undersampled Parallel MRI and single-channel Sparse MRI reconstructions. H-DSLR is a model-based deep learning approach that significantly reduces the computational complexity of Structured low-rank (SLR) algorithms. It employs both k-space and spatial domain priors.

### Relevant Papers
Pramanik, Aniket, Hemant Aggarwal, and Mathews Jacob, "Deep Generalization of Structured Low-Rank Algorithms (Deep-SLR)", IEEE Transactions on Medical Imaging, 2020. https://ieeexplore.ieee.org/document/9159672

Pramanik, Aniket, Hemant Aggarwal, and Mathews Jacob, "Calibrationless Parallel MRI using Model Based Deep Learning (C-MODL)", 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI). https://ieeexplore.ieee.org/document/9098490



## Results
<img src="brain_6x_recon.png"  title="hover text">

## Code Details
The codes have been written in python-3.7 using the Tensorflow-1.15 platform.
### Training and Testing datasets
The training dataset consists of 12-channel brain MRI images from 4 subjects collected through SSFP acquisition protocol. There are 90 slices per subject which makes it 90 x 4 = 360 slices in total. Each slice is of dimension 256 x 232 x 12. The dataset can be downloaded from the link https://drive.google.com/file/d/1Fml2PtQuECfbXAI86OYqzBb7K_CiQ2tk/view?usp=sharing .\
The testing dataset is uploaded as tst_img.npy which consists of a slice (256 x 232 x 12) from another subject unseen by the network during training.
### Description of Python scripts
1. trn_HDSLR.py : It is the training code. The H-DSLR model is trained on the 360 12-channel brain slices described above.
2. tst_HDSLR.py : It is the code for testing a pre-trained model on the test dataset uploaded as tst_img.npy. The pre-trained model is inside the directory 'savedModels'.
3. auxiliaryFunctions.py : The training and testing dataset preparation related functions are defined in this script.
4. HDSLR.py : Defines the H-DSLR network architecture to be trained.


