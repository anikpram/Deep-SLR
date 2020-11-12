#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:33:40 2020

@author: apramanik
"""

# import some libraries
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import data_processing_functions as sf
import HDSLR as mm
import scipy as sp
from scipy.io import savemat
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY

epochs=100
savemodNepoch=20
batchSize=1
K=10
nImg=256*25

#--------------------------------------------------------------------------SAME
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+ str(nImg)+'I_'+  str(epochs)+'E_'+str(batchSize)+'B'+str(K)+'K'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'

#%% read multi-channel dataset
sigma=1e4 #add noise
trnOrg,trnAtb,trnjwx,trnjwy,trnmask=sf.getDataTrng(nImg,sigma)
st=np.zeros((2,),dtype=np.float32)
#%% save test model
tf.reset_default_graph()
atbT = tf.placeholder(tf.float32,shape=(None,1,256,170,2),name='atb')
jwxT = tf.placeholder(tf.complex64,shape=(None,1,256,170,1),name='jwx')
jwyT = tf.placeholder(tf.complex64,shape=(None,1,256,170,1),name='jwy')
maskT = tf.placeholder(tf.complex64,shape=(None,1,256,170,1),name='mask')
outk=mm.makeModel(atbT,jwxT,jwyT,maskT,K)
fhatT=outk
fhatT=tf.identity(fhatT,name='fhatT')
sessFileNameTst=directory+'/modelTst'

saver=tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved:' +savedFile)
#%%
tf.reset_default_graph()
atbP = tf.placeholder(tf.float32,shape=(None,1,256,170,2),name='atb')
orgP = tf.placeholder(tf.float32,shape=(None,256,170),name='org')
jwxP = tf.placeholder(tf.complex64,shape=(None,1,256,170,1),name='jwx')
jwyP = tf.placeholder(tf.complex64,shape=(None,1,256,170,1),name='jwy')
maskP = tf.placeholder(tf.complex64,shape=(None,1,256,170,1),name='mask')
#%% creating the dataset
nTrn=trnOrg.shape[0]
nBatch= int(np.floor(np.float32(nTrn)/batchSize))
nSteps= nBatch*epochs

trnData = tf.data.Dataset.from_tensor_slices((orgP,atbP,jwxP,jwyP,maskP))
trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=trnOrg.shape[0])
trnData=trnData.batch(batchSize)
trnData=trnData.prefetch(5)
iterator=trnData.make_initializable_iterator()
orgT,atbT,jwxT,jwyT,maskT = iterator.get_next('getNext')

#%% make training model

outk=mm.makeModel(atbT,jwxT,jwyT,maskT,K)
fhatT=outk
fhatT=tf.identity(fhatT,name='fhat')
loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(fhatT-orgT), 2),axis=0))
tf.summary.scalar('lossT', loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)



#%% training code


print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSteps:',nSteps,'nSamples:',nTrn)

saver = tf.train.Saver(max_to_keep=100)
epochloss,totalLoss,ep=[],[],0
lossT1 = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT1)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,restorePath+'/model-100')
    #wts=sess.run(tf.global_variables())
    feedDict={orgP:trnOrg,atbP:trnAtb,jwxP:trnjwx,jwyP:trnjwy,maskP:trnmask}
    sess.run(iterator.initializer,feed_dict=feedDict)
    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_,_=sess.run([loss,update_ops,optimizer])
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                epochloss.append(avgTrnLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT1:avgTrnLoss})
                writer.add_summary(lossSum,ep)
                if np.remainder(ep,savemodNepoch)==0:
                    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
                totalLoss=[] #after each epoch empty the list of total loss
        except tf.errors.OutOfRangeError:
            break
    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
    writer.close()
sp.io.savemat(directory+'/epochloss.mat',mdict={'epochs':epochloss},appendmat=True)
end_time = time.time()
print ('Training completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')

#%%
