#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:28:16 2020

@author: apramanik
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data_processing_functions as sf
from tqdm import tqdm
cwd=os.getcwd()
tf.reset_default_graph()

#%% Load directory containing trained model and set other parameters
########################ksponly_jwxjwy_axial_sos####################################
subDirectory='10Jul_0305pm_6400I_100E_1B10K'
nImg=2
dispind=0
sigma=1e4 # add noise
#%%Read test dataset

_,tstOrg,tstA,tstjwx,tstjwy,tstmask,minv=sf.getTstDatag(nImg,sigma)
#%% Load existing model and feed with test data
print ('Now loading the model ...')
fhat=np.zeros((nImg,256,170),dtype=np.float32)
modelDir= cwd+'/savedModels/'+subDirectory #complete path
tf.reset_default_graph()
loadChkPoint=tf.train.latest_checkpoint(modelDir)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    jwxT=graph.get_tensor_by_name('jwx:0')
    jwyT=graph.get_tensor_by_name('jwy:0')
    maskT=graph.get_tensor_by_name('mask:0')
    fhatT=graph.get_tensor_by_name('fhatT:0')
    atbT=graph.get_tensor_by_name('atb:0')
    wts=sess.run(tf.global_variables())
    for i in tqdm(range(nImg)):
        dataDict={atbT:tstA[[i]],jwxT:tstjwx[[i]],jwyT:tstjwy[[i]],maskT:tstmask[[i]]}
        fhat[i]=sess.run(fhatT,feed_dict=dataDict)

print('Reconstruction done')
#%% Calculate PSNR
print('Now calculating the SNR (dB) values')
scale=np.sqrt(256.0*170.0)
fun= lambda x: np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x))))*scale
normOrg=tstOrg[:]
tstA=np.squeeze(tstA)
re,im=np.split(tstA,2,-1)
tstA=re+(1j*im)
tstA=np.squeeze(tstA,axis=-1)
atb=np.zeros(tstA.shape,dtype=np.complex64)
for j in range(nImg):
    atb[j,:,:]=fun(tstA[j,:,:])
normNoisy=atb[:]
normRec=fhat[:]
normOrg=np.abs(normOrg)
normNoisy=np.abs(normNoisy)
normRec=np.abs(normRec)
normError=np.abs(normOrg-normRec)
psnrNoisy=sf.myPSNR(normOrg,normNoisy)
psnrRec = sf.myPSNR(normOrg,normRec)
print ('  {0:.2f} {1:.2f} '.format(psnrNoisy.mean(),psnrRec.mean()))

print ('********************************')
normOrg=normOrg/normOrg.max()
normNoisy=normNoisy/normNoisy.max()
normRec=normRec/normRec.max()
normError=normError/normError.max()
#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, interpolation='bilinear')
ploterr= lambda x: plt.imshow(x,cmap=plt.cm.gray, interpolation='bilinear',clim=(0.0, 0.5))
plt.clf()
plt.subplot(141)
plot(np.abs(normOrg[dispind]))
plt.axis('off')
plt.title('Original')
plt.subplot(142)
plot(np.abs(normNoisy[dispind]))
plt.title('Input \n SNR='+str(psnrNoisy[dispind].round(2))+' dB' )
plt.axis('off')
plt.subplot(143)
plot(np.abs(normRec[dispind]))
plt.title('Single-channel H-DSLR \n SNR='+ str(psnrRec[dispind].round(2)) +' dB')
plt.axis('off')
plt.subplot(144)
ploterr(normError[dispind])
plt.title('Error Image')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()

