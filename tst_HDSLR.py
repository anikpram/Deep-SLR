
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:59:44 2018

@author: apramanik
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import auxiliaryFunctions as sf
from tqdm import tqdm
cwd=os.getcwd()
tf.reset_default_graph()

#%% specify model directory

subDirectory='20Feb_1229pm_360I_500E_0V_1B10K'




nImg=1
dispind=0
#%%Read the testing data 
org,_,atb,mask,std=sf.getTestingData(nImg=nImg)
std=np.expand_dims(std,axis=1)
std=np.expand_dims(std,axis=2)
std=np.expand_dims(std,axis=3)
std=np.tile(std,[1,12,256,232])
#%% Load trained model and reconstruct with it
print ('Now loading the model ...')
rec=np.empty_like(atb)
modelDir= cwd+'/savedModels/'+subDirectory #complete path
tf.reset_default_graph()
loadChkPoint=tf.train.latest_checkpoint(modelDir)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    maskT=graph.get_tensor_by_name('mask:0')
    fhatT=graph.get_tensor_by_name('fhatT:0')
    atbT=graph.get_tensor_by_name('atb:0')
    wts=sess.run(tf.global_variables())
    for i in tqdm(range(nImg)):
        dataDict={atbT:atb[[i]],maskT:mask[[i]]}
        rec[i]=sess.run(fhatT,feed_dict=dataDict)

print('Reconstruction done')
#%% postprocess the data to view results
org = sf.create_sos(org)
atb = sf.create_sos(sf.r2c(atb)*std)
recon = sf.create_sos(sf.r2c(rec)*std)
error = np.abs(org-recon)
ssimrec=np.zeros((nImg,),dtype=np.float64)
psnrRec=np.zeros((nImg,),dtype=np.float64)
psnrAtb=np.zeros((nImg,),dtype=np.float64)
for i in range(nImg):
    ssimrec[i] = sf.ssimch(org[i],recon[i])
    psnrAtb[i] = sf.psnr(org[i],atb[i])
    psnrRec[i] = sf.psnr(org[i],recon[i])

print ('  {0:.2f} {1:.2f} '.format(psnrAtb.mean(),psnrRec.mean()))

print ('********************************')
recon=recon/recon.max()
error=error/error.max()
atb=atb/atb.max()
org=org/org.max()
#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray,clim=(0.0, 0.8))
plt.clf()
plt.subplot(141)
st=50
end=220
plot(np.abs(org[dispind,st:end,:]))
plt.axis('off')
plt.title('Original')
plt.subplot(142)
plot(np.abs(atb[dispind,st:end,:]))
plt.title('Input \n PSNR='+str(psnrAtb[dispind].round(2))+' dB' )
plt.axis('off')
plt.subplot(143)
plot(np.abs(recon[dispind,st:end,:]))
plt.title('H-DSLR, PSNR='+ str(psnrRec[dispind].round(2)) +' dB')
plt.axis('off')
plt.subplot(144)
plot(error[dispind,st:end,:])
plt.title('Error Image')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()

