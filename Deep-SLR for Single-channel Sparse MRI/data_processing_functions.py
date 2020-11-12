#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:45:30 2020

@author: apramanik
"""


import time
import numpy as np
from scipy.io import loadmat
from skimage.measure import compare_psnr, compare_ssim

#%%
# get region from center of k-space
def undersample_kspace(data,underres):
    f=np.zeros(underres,dtype=np.complex64)
    res=data.shape
    if np.mod(res[0],[2]):
        a=int(np.divide([res[0]+1],[2]))
    else:
        a=int(np.divide([res[0]],[2]))
    if np.mod(res[1],[2]):
        b=int(np.divide([res[1]+1],[2]))
    else:
        b=int(np.divide([res[1]],[2]))
    if np.mod(underres[0],[2]):
        a1=int(np.divide([underres[0]-1],[2]))
        al=a-a1
        ah=a+a1+1
    else:
        a1=int(np.divide([underres[0]],[2]))
        al=a-a1
        ah=a+a1
    if np.mod(underres[1],[2]):
        b1=int(np.divide([underres[1]-1],[2]))
        bu=b-b1
        bl=b+b1+1
    else:
        b1=int(np.divide([underres[1]],[2]))
        bu=b-b1
        bl=b+b1
    for i in range(al,ah):
        for j in range(bu,bl):
            f[i-al,j-bu]=data[i,j]
    return f

# get region from center of k-space
def low_res_kspace(data,underres):
    f=np.zeros(data.shape,dtype=np.complex64)
    res=data.shape
    if np.mod(res[0],[2]):
        a=int(np.divide([res[0]+1],[2]))
    else:
        a=int(np.divide([res[0]],[2]))
    if np.mod(res[1],[2]):
        b=int(np.divide([res[1]+1],[2]))
    else:
        b=int(np.divide([res[1]],[2]))
    if np.mod(underres[0],[2]):
        a1=int(np.divide([underres[0]-1],[2]))
        al=a-a1
        ah=a+a1+1
    else:
        a1=int(np.divide([underres[0]],[2]))
        al=a-a1
        ah=a+a1
    if np.mod(underres[1],[2]):
        b1=int(np.divide([underres[1]-1],[2]))
        bu=b-b1
        bl=b+b1+1
    else:
        b1=int(np.divide([underres[1]],[2]))
        bu=b-b1
        bl=b+b1
    for i in range(al,ah):
        for j in range(bu,bl):
            f[i,j]=data[i,j]
    return f

# get k-space indices
def get_kspace_inds(res):
    if np.mod(res[0],[2]):
        indx=list(np.arange(0,((res[1]-1)/2)+1))
        indx=indx+list(np.arange(-(res[1]-1)/2,0))
    else:
        indx=list(np.arange((-res[1]/2),(res[1]/2)))
        indx = np.fft.ifftshift(indx)
        
    if np.mod(res[1],[2]):
        indy=list(np.arange(0,((res[0]-1)/2)+1))
        indy=indy+list(np.arange(-(res[0]-1)/2,0))
 
    else:
        indy=list(np.arange((-res[0]/2),(res[0]/2)))
        indy = np.fft.ifftshift(indy)
 

    kx,ky=np.meshgrid(indx,indy)
    kx=np.fft.fftshift(kx)
    ky=np.fft.fftshift(ky)
    kx=np.reshape(kx,(np.size(kx),1))
    ky=np.reshape(ky,(np.size(ky),1))
    return kx,ky

# compute gradient of k-space along x and y axes
def compute_gradient(data):
    res=data.shape
    a,b=get_kspace_inds(np.asarray(res))
    a=a.astype(np.float32)
    b=b.astype(np.float32)
    s=np.asarray([2,res[0],res[1]])
    dz=np.zeros(s,dtype=np.complex64)
    dz[0,:,:]=np.reshape(1j*2*a/res[1],res)
    dz[1,:,:]=np.reshape(1j*2*b/res[0],res)
    M=np.zeros(s,dtype=np.complex64)
    M[0,:,:]=data*dz[0,:,:]
    M[1,:,:]=data*dz[1,:,:]
    return M

#define x and y gradient matrices for k-space 
def jwxjwy(res):
    a,b=get_kspace_inds(np.asarray(res))
    a=a.astype(np.float32)
    b=b.astype(np.float32)  
    a=2*a/256
    b=2*b/256  
    s=np.asarray([2,res[0],res[1]])
    dz=np.zeros(s,dtype=np.complex64)
    dz[0,:,:]=np.reshape(1j*a,res)
    dz[1,:,:]=np.reshape(1j*b,res)
    return dz

# define sum of squared x and y gradient matrices for k-space
def Glhsm(res):
    a,b=get_kspace_inds(np.asarray(res))
    a=a.astype(np.float32)
    b=b.astype(np.float32)  
    a=2*a/256
    b=2*b/256   
    s=np.asarray([2,res[0],res[1]])
    dz=np.zeros(s,dtype=np.complex64)
    dz1=np.zeros(res,dtype=np.complex64)
    dz[0,:,:]=np.reshape(a*a,res)
    dz[1,:,:]=np.reshape(b*b,res)
    dz1=np.add(dz[0,:,:],dz[1,:,:])
    return dz1
#%%
# divide numpy arrays
def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c

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
#normalize image
def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        #img2[i]=div0(img[i]-img[i].min(),img[i].ptp())
        img2[i]=(img[i]-img[i].min())/(np.abs(img[i].max()-img[i].min()))
    return np.squeeze(img2).astype(img.dtype)



#%%
# normalize image
def normalize02(img):
    """
    Normalize the image between -1 and 1
    """
    img2=img/img.max()
    return img2.astype(img.dtype)
#%% normalize image
def normalizeg(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==4:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        img2[i,0]=div0(img[i,0]-img[i,0].min(),img[i,0].ptp())
        img2[i,1]=div0(img[i,1]-img[i,1].min(),img[i,1].ptp())
    return np.squeeze(img2).astype(img.dtype)
#%%
# compute PSNR
def myPSNR(org,recon):
    error=np.abs(org-recon)**2
    N=np.prod(org.shape[-2:])
    mse=np.sum(error,axis=(-1,-2))/N
    maxval=np.abs(np.max(org,axis=(-1,-2)))
    psnr=20*np.log10(maxval/(np.sqrt(mse)+1e-10 ))
    return psnr

# compute SNR
def mySNR(org,recon):
    error=np.abs(org-recon)**2
    N=np.prod(org.shape[-2:])
    mse=np.sum(error,axis=(-1,-2))/N
    orig=np.abs(org)**2
    power=np.sum(orig,axis=(-1,-2))/N
    snr=20*np.log10(np.sqrt(power)/np.sqrt(mse))
    return snr

# compute mean squared error
def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)

# compute normalized mean squared error
def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

# compute PSNR
def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())

# compute SSIM
def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )

# compute SSIM
def ssimch(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt, pred, multichannel=True, data_range=gt.max()
    )
    
    
#%% Process training datasets
def getDataTrng(nImg=256*25,sigma=1,res=[256,170]):
    scale = np.sqrt(256.0*170.0)
    a=jwxjwy(res)
    jwx=a[0]
    jwy=a[1]
    del a
    jwx=np.tile(jwx,[nImg,1,1])
    jwx=np.expand_dims(jwx,-1)
    jwx=np.expand_dims(jwx,1)
    jwy=np.tile(jwy,[nImg,1,1])
    jwy=np.expand_dims(jwy,-1)
    jwy=np.expand_dims(jwy,1)
    org=np.load('train_25_subjects_axial.npy')
    org=org[0:nImg]
    for i in range(nImg):
        org[i,:,:] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(org[i,:,:])))
    org = org/scale
    m=loadmat('mask4f_256by170.mat')['a']
    mask=np.tile(m,[nImg,1,1])
    mask=mask.astype(np.complex64)
    orgk,atb,_=generateUndersampledomodl(org,mask)
    mask=np.expand_dims(mask,-1)
    mask=np.expand_dims(mask,1)
    atb=c2romodl(atb,nImg)
    for i in range(nImg):
        orgk[i,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(orgk[i,:,:])))
    orgk = np.abs(orgk)*scale
    return orgk,atb,jwx,jwy,mask




# Process testing datasets
def getTstDatag(nImg=256*10,sigma=1,res=[256,170]):
    scale = np.sqrt(256.0*170.0)
    a=jwxjwy(res)
    jwx=a[0]
    jwy=a[1]
    del a
    jwx=np.tile(jwx,[nImg,1,1])
    jwx=np.expand_dims(jwx,-1)
    jwx=np.expand_dims(jwx,1)
    jwy=np.tile(jwy,[nImg,1,1])
    jwy=np.expand_dims(jwy,-1)
    jwy=np.expand_dims(jwy,1)
    #org=np.load('test_10_subjects_axial.npy')
    org=np.load('test_2_img_axial.npy')
    org=org[0:nImg]
    for i in range(nImg):
        org[i,:,:] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(org[i,:,:])))
    org = org/scale
    m=loadmat('mask4f_256by170.mat')['a']
    mask=np.tile(m,[nImg,1,1])
    mask=mask.astype(np.complex64)
    orgk,atb,minv=generateUndersampledomodl(org,mask)
    mask=np.expand_dims(mask,-1)
    mask=np.expand_dims(mask,1)
    atb=c2romodl(atb,nImg)
    for i in range(nImg):
        orgk[i,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(orgk[i,:,:])))
    orgk = np.abs(orgk)*scale
    return org,orgk,atb,jwx,jwy,mask,minv


#%%
# create A operator 
def piAomodl(x,mask,nrow,ncol):
    """ This is a the A operator as defined in the paper"""
    kspace=np.reshape(x,(nrow,ncol) )
    res=kspace[mask!=0]
    return kspace,res

# create At operator
def piAtomodl(kspaceUnder,mask,nrow,ncol):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((nrow,ncol),dtype=np.complex64)
    temp[mask!=0]=kspaceUnder
    minv=np.std(temp)
    temp=temp/minv
    return temp,minv

# generate undersampled k-space
def generateUndersampledomodl(org,mask):
    nSlice,nrow,ncol=org.shape
    orgk=np.empty(org.shape,dtype=np.complex64)
    atb=np.empty(org.shape,dtype=np.complex64)
    minv=np.zeros((nSlice,),dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: piAomodl(z,mask[i],nrow,ncol)
        At = lambda z: piAtomodl(z,mask[i],nrow,ncol)
        orgk[i],orgkus=A(org[i])
        y=orgkus[:]
        atb[i],minv[i]=At(y)
        orgk[i]=orgk[i]/minv[i]
    del org    
    return orgk,atb,minv

#%% convert 2-channel real data to complex data
def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros( inp.shape[0:3],dtype=dtype)
    out=inp[...,0]+1j*inp[...,1]
    return out

# convert complex data to 2-channel real data
def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros( inp.shape+(2,),dtype=dtype)
    out[...,0]=inp.real
    out[...,1]=inp.imag
    return out

# convert complex data to 2-channel real data
def c2romodl(inp,nImg):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros((nImg,1,256,170,2),dtype=dtype)
    out[:,0,:,:,0]=np.real(inp)
    out[:,0,:,:,1]=np.imag(inp)
    return out
