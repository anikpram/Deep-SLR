"""
Created on Aug 6th, 2018

This file contains some supporting functions used during training and testing.

@author:Aniket
"""
import time
import numpy as np
from scipy.io import loadmat
from skimage.measure import compare_psnr, compare_ssim

#%% This provide functionality similar to matlab's tic() and toc()
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)



#%%

def create_sos(ksp):
    nImg,nCh,nrow,ncol = ksp.shape
    scale = np.sqrt(float(nrow)*float(ncol))
    fun = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    for i in range(nImg):
        for j in range(nCh):
            ksp[i,j,:,:]=fun(ksp[i,j,:,:])
    sos = np.sqrt(np.sum(np.square(np.abs(ksp)*scale),axis=1))
    return sos
    


#%%

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )

def ssimch(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt, pred, multichannel=True, data_range=gt.max()
    )


#%% Here I am reading the dataset for training and testing from dataset.hdf5 file




def getData(nImg=360):
    print('Reading data')
    tic()
    org=np.load('trn_data_90im_4_subjects.npy')
    mask=loadmat('vardenmask_6f.mat')['b']
    mask=mask.astype(np.complex64)
    mask=np.tile(mask,[nImg,1,1])
    toc()
    print('Undersampling')
    tic()
    orgk,atb,_=generateUndersampled(org,mask)
    atb=c2r(atb)
    orgk=c2r(orgk)
    mask=np.expand_dims(mask,axis=1)
    mask=np.tile(mask,[1,12,1,1])
    mask=np.expand_dims(mask,axis=-1)
    orgk=np.expand_dims(orgk,axis=-1)
    atb=np.expand_dims(atb,axis=-1)
    toc()
    print('Data prepared!')
    return orgk,atb,mask





def getTestingData(nImg=90):
    print('Reading data')
    tic()
    org=np.load('tst_data_90im_1_subject.npy')
    mask=loadmat('vardenmask_6f.mat')['b']
    mask=mask.astype(np.complex64)
    mask=np.tile(mask,[nImg,1,1])
    toc()
    print('Undersampling')
    tic()
    orgk,atb,minv=generateUndersampled(org,mask)
    atb=c2r(atb)
    orgk=c2r(orgk)
    mask=np.expand_dims(mask,axis=1)
    mask=np.tile(mask,[1,12,1,1])
    mask=np.expand_dims(mask,axis=-1)
    orgk=np.expand_dims(orgk,axis=-1)
    atb=np.expand_dims(atb,axis=-1)
    toc()
    print('Data prepared!')
    return org,orgk,atb,mask,minv




#%%

def usp(x,mask,nrow,ncol,ncoil):
    """ This is a the A operator as defined in the paper"""
    kspace=np.reshape(x,(ncoil,nrow,ncol) )
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))
    res=kspace[mask!=0]
    return kspace,res

def usph(kspaceUnder,mask,nrow,ncol,ncoil):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((ncoil,nrow,ncol),dtype=np.complex64)
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))
    temp[mask!=0]=kspaceUnder
    minv=np.std(temp)
    temp=temp/minv
    return temp,minv

def generateUndersampled(org,mask):
    nSlice,ncoil,nrow,ncol=org.shape
    orgk=np.empty(org.shape,dtype=np.complex64)
    atb=np.empty(org.shape,dtype=np.complex64)
    minv=np.zeros((nSlice,),dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: usp(z,mask[i],nrow,ncol,ncoil)
        At = lambda z: usph(z,mask[i],nrow,ncol,ncoil)
        orgk[i],y=A(org[i])
        atb[i],minv[i]=At(y)
        orgk[i]=orgk[i]/minv[i]
    del org    
    return orgk,atb,minv



#%%
def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    inp=np.squeeze(inp,axis=-1)
    nImg,nCh,nrow,ncol=inp.shape
    out=np.zeros((nImg,nCh,nrow,ncol),dtype=dtype)
    re,im=np.split(inp,2,axis=1)
    out=re+(1j*im)
    return out

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    nImg,nCh,nrow,ncol=inp.shape
    out=np.zeros((nImg,nCh*2,nrow,ncol),dtype=dtype)
    out[:,0:nCh,:,:]=np.real(inp)
    out[:,nCh:nCh*2,:,:]=np.imag(inp)
    return out

