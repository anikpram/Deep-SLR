#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:01:13 2018

@author: apramanik
"""




import tensorflow as tf




def convLayer(x, szW,i):
    """
    create a convolution layer.
    """
    with tf.name_scope('layers'):
        with tf.variable_scope('lay'+str(i)):
            W=tf.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
            y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME',data_format='NCHW')
            if i!=5 and i!=10:
                y=tf.nn.relu(y)
    return y

def r2c(x):
    re,im=tf.split(x,[12,12],1)
    x=tf.complex(re,im)
    return x

def c2r(x):
    x=tf.concat([tf.real(x),tf.imag(x)],1)
    return x



def fivelim(inp):
    """
    spatial network
    """
    scale=tf.complex(tf.sqrt(256.0*232.0),0.0)
    inp=r2c(inp)
    inp=tf.squeeze(inp,axis=-1)
    inpi=tf.ifft2d(inp)
    inpi=scale*inpi
    inpi=c2r(inpi)
    with tf.name_scope('imgntw'):
        x=convLayer(inpi,(3,3,24,64),1)
        x=convLayer(x,(3,3,64,64),2)
        x=convLayer(x,(3,3,64,64),3)
        x=convLayer(x,(3,3,64,64),4)
        x=convLayer(x,(3,3,64,24),5)
        x=inpi+x
    x=r2c(x)
    x=tf.fft2d(x)
    x=x/scale
    x=tf.expand_dims(x,axis=-1)
    x=c2r(x)
    return x


def fivelk(inp):
    """
    kspace network.
    """
    inp=tf.squeeze(inp,axis=-1)
    with tf.name_scope('kspntw'):
        x=convLayer(inp,(3,3,24,64),6)
        x=convLayer(x,(3,3,64,64),7)
        x=convLayer(x,(3,3,64,64),8)
        x=convLayer(x,(3,3,64,64),9)
        x=convLayer(x,(3,3,64,24),10)
        x=inp+x
    x=tf.expand_dims(x,axis=-1)
    return x


def dc(rhs, mask, lam1,lam2):
    """
    The data consistency block.
    """
    lam1=tf.complex(lam1,0.)
    lam2=tf.complex(lam2,0.)
    lhs=mask+lam1+lam2
    rhs=r2c(rhs)
    output=tf.div(rhs,lhs)
    output=c2r(output)
    return output

def getLambdak():
    """
    create a shared variable called lambda.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam2', dtype=tf.float32, initializer=1.0)
    return lam


def getLambdaim():
    """
    create a shared variable called lambda.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=1.0)
    return lam


def makeModel(atb,mask,K):
    """
    This is the main function that creates the model.

    """
    out={}
    out['dc0']=atb
    with tf.name_scope('HDSLR'):
        with tf.variable_scope('Wts',reuse=tf.AUTO_REUSE):
            for i in range(1,K+1): # recursive network with shared parameters across iterations
                j=str(i)
                out['dwim'+j]=fivelim(out['dc'+str(i-1)]) # spatial network output
                out['dwk'+j]=fivelk(out['dc'+str(i-1)]) # kspace network output
                lam1=getLambdaim()
                lam2=getLambdak()
                rhs=atb + lam1*out['dwim'+j] + lam2*out['dwk'+j] 
                out['dc'+j]=dc(rhs,mask,lam1,lam2) # data consitency block output            
    return out

