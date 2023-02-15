#!/usr/bin/env python
# coding: utf-8

import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' ## enable xla devices
#get_ipython().system('nvidia-smi ## check Nvidia GPU details')
# !nvidia-smi -L
# !nvidia-smi -q
#get_ipython().system('lscpu')
# !lscpu |grep 'Model name' ## Check CPU details
# !lscpu |grep 'Number of Socket(s):'
# !lscpu | grep 'Core(s) each processor has/per socket:'
# !lscpu | grep 'Number of threads/core:'
#get_ipython().system("free -h --si | awk  '/Mem:/{print $2}'")

# if use tpu
#import tensorflow as tf
# try:
#   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
#   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
# except ValueError:
#   raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

#tf.config.experimental_connect_to_cluster(tpu)
#tf.tpu.experimental.initialize_tpu_system(tpu)
#tpu_strategy = tf.distribute.TPUStrategy(tpu)

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

from tensorflow.python import ipu
config= ipu.config.IPUConfig()
config.auto_select_ipus=1
config.configure_ipu_system()

# Common imports
import numpy as np
import os

# To plot pretty figures
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=18)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

import math
import time 
from mpl_toolkits.mplot3d import Axes3D


tf.keras.backend.set_floatx('float32')
dtype = tf.float32
dtype_np = np.float32

# # Functions linking to the AI libraries
# Keras layer to replace the Eager-Mode function we had before
# since we cannot create tf.Variables inside a (repeated) function call in graph mode
class BoundaryConditionVelocity(keras.layers.Layer):
    def __init__(self, nx, ub, input_shape):
        super(BoundaryConditionVelocity, self).__init__()
        self.nx = nx
        self.ub = ub
        self.inp_shape = [1,*input_shape]
        self.tempu = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)
        self.tempv = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)
        self.tempw = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)

    def call(self, values_u, values_v,values_w):
        self.tempu.assign(values_u)
        self.tempv.assign(values_v)
        self.tempw.assign(values_w)
        
        self.tempu[0,:,:,0,0].assign(tf.ones((1,self.nx,self.nx), dtype=dtype)[0,:]*self.ub)
        self.tempv[0,:,:,0,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])
        self.tempw[0,:,:,0,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])

        self.tempu[0,:,:,nx-1,0].assign(tf.ones((1,self.nx,self.nx), dtype=dtype)[0,:]*self.ub)
        self.tempv[0,:,:,nx-1,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])
        self.tempw[0,:,:,nx-1,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])
        
        self.tempu[0,:,0,:,0].assign(values_u[0,:,1,:,0])
        self.tempv[0,:,0,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])
        self.tempw[0,:,0,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])

        self.tempu[0,:,nx-1,:,0].assign(values_u[0,:,self.nx-2,:,0])
        self.tempv[0,:,nx-1,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])  
        self.tempw[0,:,nx-1,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])
        
        self.tempu[0,0,:,:,0].assign(values_u[0,1,:,:,0])
        self.tempv[0,0,:,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])
        self.tempw[0,0,:,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])
        
        self.tempu[0,nx-1,:,:,0].assign(values_u[0,self.nx-2,:,:,0])   
        self.tempv[0,nx-1,:,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:])   
        self.tempw[0,nx-1,:,:,0].assign(tf.zeros((1,self.nx,self.nx), dtype=dtype)[0,:]) 

        return self.tempu,self.tempv,self.tempw


# Keras layer to replace the Eager-Mode function we had before
# since we cannot create tf.Variables inside a (repeated) function call in graph mode
class BoundaryConditionPressure(keras.layers.Layer):
    def __init__(self, nx, input_shape):
        super(BoundaryConditionPressure, self).__init__()
        self.nx = nx
        self.inp_shape = [1,*input_shape]
        self.tempp = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)

    def call(self, values_p):
        self.tempp.assign(values_p)

        self.tempp[0,:,:,nx-1,0].assign(tf.zeros((1,nx,nx), dtype=dtype)[0,:]) 
        self.tempp[0,:,:,0,0].assign(values_p[0,:,:,1,0])
        
        self.tempp[0,:,0,:,0].assign(values_p[0,:,1,:,0])     
        self.tempp[0,:,nx-1,:,0].assign(values_p[0,:,nx-2,:,0]) 
        
        self.tempp[0,0,:,:,0].assign(values_p[0,1,:,:,0])     
        self.tempp[0,nx-1,:,:,0].assign(values_p[0,nx-2,:,:,0])  
        
        return self.tempp


# Keras layer to replace the Eager-Mode function we had before
# since we cannot create tf.Variables inside a (repeated) function call in graph mode
class BoundaryConditionSource(keras.layers.Layer):
    'Define inflow boundary conditions for source terms to'
    'avoid incorrect paddings caused by CNNs'
    def __init__(self, nx, input_shape):
        super(BoundaryConditionSource, self).__init__()
        self.nx = nx  
        self.inp_shape = [1,*input_shape]
        self.tempb = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)

    def call(self, b):
        self.tempb.assign(b)   
        self.tempb[0,:,:,0,0].assign(b[0,:,:,1,0])
    
        return self.tempb  
        # return tf.convert_to_tensor(b)


# Keras layer to replace the Eager-Mode function we had before
# since we cannot create tf.Variables inside a (repeated) function call in graph mode
class BluffBody(keras.layers.Layer):
    def __init__(self, sigma, dt, xmin, xmax, ymin, ymax, zmin, zmax, input_shape):
        super(BluffBody, self).__init__()
        self.sigma = sigma
        self.dt = dt
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.inp_shape = [1,*input_shape]
        self.temp1 = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)
        self.temp2 = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)
        self.temp3 = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)
        
    def call(self, values_u, values_v, values_w):
        self.temp1.assign(values_u)
        self.temp2.assign(values_v)
        self.temp3.assign(values_w)

        self.temp1[0,self.zmin:self.zmax,self.ymin:self.ymax,self.xmin:self.xmax,0].assign(self.temp1[0,self.zmin:self.zmax,self.ymin:self.ymax,self.xmin:self.xmax,0]/(1+self.dt*self.sigma))
        self.temp2[0,self.zmin:self.zmax,self.ymin:self.ymax,self.xmin:self.xmax,0].assign(self.temp2[0,self.zmin:self.zmax,self.ymin:self.ymax,self.xmin:self.xmax,0]/(1+self.dt*self.sigma))
        self.temp3[0,self.zmin:self.zmax,self.ymin:self.ymax,self.xmin:self.xmax,0].assign(self.temp3[0,self.zmin:self.zmax,self.ymin:self.ymax,self.xmin:self.xmax,0]/(1+self.dt*self.sigma))
        
        return self.temp1,self.temp2,self.temp3


# Keras layer to replace the Eager-Mode function we had before
# since we cannot create tf.Variables inside a (repeated) function call in graph mode
class MakeTensorZeroInTheMiddle(keras.layers.Layer):
    def __init__(self, nx, input_shape):
        super(MakeTensorZeroInTheMiddle, self).__init__()
        self.inp_shape = input_shape
        self.nx = nx
        self.tempr = tf.Variable(tf.ones(self.inp_shape, dtype=dtype), trainable=False)
    
    def call(self, r):
        self.tempr.assign(r)
        self.tempr[0,:,:,nx-1,0].assign(tf.zeros((1,nx,nx), dtype=dtype)[0,:])

        return self.tempr


#@tf.function(jit_compile=True)
def save_data(values_u,values_v,values_w,values_p,n_out,itime):
    if itime % n_out == 0:  
        np.save("result_3d_27point/u"+str(itime), arr=values_u[0,:,:,:,0])
        np.save("result_3d_27point/v"+str(itime), arr=values_v[0,:,:,:,0])
        np.save("result_3d_27point/w"+str(itime), arr=values_w[0,:,:,:,0])
        np.save("result_3d_27point/p"+str(itime), arr=values_p[0,:,:,:,0])

#import tensorflow as tf
#tf.config.list_physical_devices()
#is_gpu = len(tf.config.list_physical_devices('GPU')) > 0 
#print(is_gpu)


# Create Keras model here.
# The model is not a Sequential type, since we have a more complex setup regarding
# which tensors go to/from which layers etc
def create_model(nx,ny,nz):
    multi_itr = 10
    mgsolver = True
    j_itr = 1

    dt = 0.1
    dx = 1
    Re = 1/5     
    ub = 1
    sigma = 100000  # bluff body
    # nx = 32
    # ny = 32
    # nz = 32
    ratio = int(nx/nz)
    nlevel = int(math.log(nz, 2)) + 1 
    print('Levels of Multigrid:', nlevel)
    print('Aspect ratio of Domain:', ratio)


    # # Weights of CNNs layers

    pd1 = [[2/26, 3/26,  2/26],
        [3/26, 6/26,  3/26],
        [2/26, 3/26,  2/26]]
    pd2 = [[3/26, 6/26,  3/26],
        [6/26, -88/26, 6/26],
        [3/26, 6/26,  3/26]]
    pd3 = [[2/26, 3/26,  2/26],
        [3/26, 6/26,  3/26],
        [2/26, 3/26,  2/26]]
    w1 = np.zeros([1,3,3,3,1], dtype=dtype_np)
    w1[0,0,:,:,0] = np.array(pd1)*dt*Re/dx**2
    w1[0,1,:,:,0] = np.array(pd2)*dt*Re/dx**2 
    w1[0,2,:,:,0] = np.array(pd3)*dt*Re/dx**2 


    p_div_x1 = [[-0.014, 0.0, 0.014],
        [-0.056, 0.0, 0.056],
        [-0.014, 0.0, 0.014]]
    p_div_x2 = [[-0.056, 0.0, 0.056],
        [-0.22, 0.0, 0.22],
        [-0.056, 0.0, 0.056]]
    p_div_x3 = [[-0.014, 0.0, 0.014],
        [-0.056, 0.0, 0.056],
        [-0.014, 0.0, 0.014]]

    p_div_y1 = [[0.014, 0.056, 0.014],
        [0.0, 0.0, 0.0],
        [-0.014, -0.056, -0.014]]
    p_div_y2 = [[0.056, 0.22, 0.056],
        [0.0, 0.0, 0.0],
        [-0.056, -0.22, -0.056]]
    p_div_y3 = [[0.014, 0.056, 0.014],
        [0.0, 0.0, 0.0],
        [-0.014, -0.056, -0.014]]

    p_div_z1 = [[0.014, 0.056, 0.014],
        [0.056, 0.22, 0.056],
        [0.014, 0.056, 0.014]]
    p_div_z2 = [[0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]]
    p_div_z3 = [[-0.014, -0.056, -0.014],
        [-0.056, -0.22, -0.056],
        [-0.014, -0.056, -0.014]]
    w2 = np.zeros([1,3,3,3,1], dtype=dtype_np)
    w3 = np.zeros([1,3,3,3,1], dtype=dtype_np)
    w4 = np.zeros([1,3,3,3,1], dtype=dtype_np)

    w2[0,0,:,:,0] = np.array(p_div_x1)*ub*dt
    w2[0,1,:,:,0] = np.array(p_div_x2)*ub*dt 
    w2[0,2,:,:,0] = np.array(p_div_x3)*ub*dt 

    w3[0,0,:,:,0] = np.array(p_div_y1)*ub*dt
    w3[0,1,:,:,0] = np.array(p_div_y2)*ub*dt
    w3[0,2,:,:,0] = np.array(p_div_y3)*ub*dt 

    w4[0,0,:,:,0] = np.array(p_div_z1)*ub*dt 
    w4[0,1,:,:,0] = np.array(p_div_z2)*ub*dt
    w4[0,2,:,:,0] = np.array(p_div_z3)*ub*dt


    pA1 = [[2/26, 3/26,  2/26],
        [3/26, 6/26,  3/26],
        [2/26, 3/26,  2/26]]
    pA2 = [[3/26, 6/26,  3/26],
        [6/26, -88/26, 6/26],
        [3/26, 6/26,  3/26]]
    pA3 = [[2/26, 3/26,  2/26],
        [3/26, 6/26,  3/26],
        [2/26, 3/26,  2/26]]
    w5 = np.zeros([1,3,3,3,1], dtype=dtype_np)
    w5[0,0,:,:,0] = -np.array(pA1)/dx**2
    w5[0,1,:,:,0] = -np.array(pA2)/dx**2 
    w5[0,2,:,:,0] = -np.array(pA3)/dx**2 


    pctyu1 = [[-0.014, 0.0, 0.014],
        [-0.056, 0.0, 0.056],
        [-0.014, 0.0, 0.014]]
    pctyu2 = [[-0.056, 0.0, 0.056],
        [-0.22, 0.0, 0.22],
        [-0.056, 0.0, 0.056]]
    pctyu3 = [[-0.014, 0.0, 0.014],
        [-0.056, 0.0, 0.056],
        [-0.014, 0.0, 0.014]]

    pctyv1 = [[0.014, 0.056, 0.014],
        [0.0, 0.0, 0.0],
        [-0.014, -0.056, -0.014]]
    pctyv2 = [[0.056, 0.22, 0.056],
        [0.0, 0.0, 0.0],
        [-0.056, -0.22, -0.056]]
    pctyv3 = [[0.014, 0.056, 0.014],
        [0.0, 0.0, 0.0],
        [-0.014, -0.056, -0.014]]

    pctyw1 = [[0.014, 0.056, 0.014],
        [0.056, 0.22, 0.056],
        [0.014, 0.056, 0.014]]
    pctyw2 = [[0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]]
    pctyw3 = [[-0.014, -0.056, -0.014],
        [-0.056, -0.22, -0.056],
        [-0.014, -0.056, -0.014]]
    w6 = np.zeros([1,3,3,3,1], dtype=dtype_np)
    w7 = np.zeros([1,3,3,3,1], dtype=dtype_np)
    w8 = np.zeros([1,3,3,3,1], dtype=dtype_np)
    w9 = np.zeros([1,2,2,2,1], dtype=dtype_np)
    w6[0,0,:,:,0] = np.array(pctyu1)/dt
    w6[0,1,:,:,0] = np.array(pctyu2)/dt 
    w6[0,2,:,:,0] = np.array(pctyu3)/dt 
    w7[0,0,:,:,0] = np.array(pctyv1)/dt 
    w7[0,1,:,:,0] = np.array(pctyv2)/dt 
    w7[0,2,:,:,0] = np.array(pctyv3)/dt 
    w8[0,0,:,:,0] = np.array(pctyw1)/dt 
    w8[0,1,:,:,0] = np.array(pctyw2)/dt 
    w8[0,2,:,:,0] = np.array(pctyw3)/dt 
    w9[0,:,:,:,0] = 0.125

    kernel_initializer_1 = tf.keras.initializers.constant(w1)
    kernel_initializer_2 = tf.keras.initializers.constant(w2)
    kernel_initializer_3 = tf.keras.initializers.constant(w3)
    kernel_initializer_4 = tf.keras.initializers.constant(w4)
    kernel_initializer_5 = tf.keras.initializers.constant(w5)
    kernel_initializer_6 = tf.keras.initializers.constant(w6)
    kernel_initializer_7 = tf.keras.initializers.constant(w7)
    kernel_initializer_8 = tf.keras.initializers.constant(w8)
    kernel_initializer_9 = tf.keras.initializers.constant(w9)
    bias_initializer = tf.keras.initializers.constant(np.zeros((1,), dtype=dtype_np))

    input_shape = (nx,ny,nz,1)
    values_u_inp = keras.Input(shape=input_shape)
    values_v_inp = keras.Input(shape=input_shape)
    values_w_inp = keras.Input(shape=input_shape)
    values_p_inp = keras.Input(shape=input_shape)

    values_u,values_v,values_w = BoundaryConditionVelocity(nx, ub, input_shape)(values_u_inp,values_w_inp,values_v_inp)
    values_p = BoundaryConditionPressure(nx, input_shape)(values_p_inp)

    # Momentum equation              
    CNN3D_central_2nd_dif = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                        kernel_initializer=kernel_initializer_1,
                                        bias_initializer=bias_initializer)

    CNN3D_central_2nd_xadv = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_2,
                                    bias_initializer=bias_initializer)

    CNN3D_central_2nd_yadv = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_3,
                                    bias_initializer=bias_initializer)

    CNN3D_central_2nd_zadv = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_4,
                                    bias_initializer=bias_initializer)

    CNN3D_Su = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_6,
                                    bias_initializer=bias_initializer)

    CNN3D_Sv = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_7,
                                    bias_initializer=bias_initializer)

    CNN3D_Sw = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_8,
                                    bias_initializer=bias_initializer)


    CNN3D_pu = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_2,
                                    bias_initializer=bias_initializer)

    CNN3D_pv = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_3,
                                    bias_initializer=bias_initializer)

    CNN3D_pw = keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                                    kernel_initializer=kernel_initializer_4,
                                    bias_initializer=bias_initializer)

    # # # Libraries for multigrid algorithms
    layers_multigrid = {}
    for i in range(nlevel):
        layers_multigrid['CNN3D_A_'+str(2**i)] = tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         
                                                                        kernel_initializer=kernel_initializer_5,
                                                                        bias_initializer=bias_initializer)


    for i in range(nlevel-1):
        layers_multigrid['CNN3D_res_'+str(2**(i+1))] = tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                                                              kernel_initializer=kernel_initializer_9,
                                                                              bias_initializer=bias_initializer)  


    for i in range(nlevel-1):
        layers_multigrid['CNN3D_prol_'+str(2**i)] = tf.keras.layers.UpSampling3D(size=(2, 2, 2))

    
    a_u = CNN3D_central_2nd_dif(values_u) -     values_u*CNN3D_central_2nd_xadv(values_u) -     values_v*CNN3D_central_2nd_yadv(values_u) -     values_w*CNN3D_central_2nd_zadv(values_u)
    b_u = 0.5*a_u + values_u
    a_v = CNN3D_central_2nd_dif(values_v) -     values_u*CNN3D_central_2nd_xadv(values_v) -     values_v*CNN3D_central_2nd_yadv(values_v) -     values_w*CNN3D_central_2nd_zadv(values_v)
    b_v = 0.5*a_v + values_v
    a_w = CNN3D_central_2nd_dif(values_w) -     values_u*CNN3D_central_2nd_xadv(values_w) -     values_v*CNN3D_central_2nd_yadv(values_w) -     values_w*CNN3D_central_2nd_zadv(values_w)
    b_w = 0.5*a_w + values_w
    
    c_u = CNN3D_central_2nd_dif(b_u) -     b_u*CNN3D_central_2nd_xadv(b_u) -     b_v*CNN3D_central_2nd_yadv(b_u) -     b_w*CNN3D_central_2nd_zadv(b_u)
    
    c_v = CNN3D_central_2nd_dif(b_v) -     b_u*CNN3D_central_2nd_xadv(b_v) -     b_v*CNN3D_central_2nd_yadv(b_v) -     b_w*CNN3D_central_2nd_zadv(b_v)

    c_w = CNN3D_central_2nd_dif(b_w) -     b_u*CNN3D_central_2nd_xadv(b_w) -     b_v*CNN3D_central_2nd_yadv(b_w) -     b_w*CNN3D_central_2nd_zadv(b_w)
    
    values_u = values_u + c_u    
    values_v = values_v + c_v
    values_w = values_w + c_w
    # IBM 
    values_u,values_v,values_w = BluffBody(sigma,dt,11,21,27,37,27,37,input_shape)(values_u,values_v,values_w)
    # gradp
    values_u = values_u - CNN3D_pu(values_p)
    values_v = values_v - CNN3D_pv(values_p)  
    values_w = values_w - CNN3D_pw(values_p)
    values_u,values_v,values_w = BoundaryConditionVelocity(nx, ub, input_shape)(values_u,values_w,values_v)
    # possion equation (multi-grid) A*P = Su
    b = -(CNN3D_Su(values_u) + CNN3D_Sv(values_v) + CNN3D_Sw(values_w))
    b = BoundaryConditionSource(nx, input_shape)(b)
    if mgsolver == True:
        for multi_grid in range(multi_itr): 
            w_1 = tf.zeros([1,1,1,1,1], dtype=dtype)
            r_init = layers_multigrid["CNN3D_A_32"](values_p) - b   
            r = MakeTensorZeroInTheMiddle(nx, r_init.shape)(r_init)
            # r[0,:,:,nx-1,0].assign(tf.zeros((1,nx,nx))[0,:]) 
            # r_32 = layers_multigrid["CNN3D_res_64"](r) 
            r_16 = layers_multigrid["CNN3D_res_32"](r)
            r_8 = layers_multigrid["CNN3D_res_16"](r_16)
            r_4 = layers_multigrid["CNN3D_res_8"](r_8)
            r_2 = layers_multigrid["CNN3D_res_4"](r_4)
            r_1 = layers_multigrid["CNN3D_res_2"](r_2)  
            for Jacobi in range(j_itr):
                w_1 = (w_1 - layers_multigrid["CNN3D_A_1"](w_1)/w5[0,1,1,1,0] + r_1/w5[0,1,1,1,0])
            w_2 = layers_multigrid["CNN3D_prol_1"](w_1)             
            for Jacobi in range(j_itr):
                w_2 = (w_2 - layers_multigrid["CNN3D_A_2"](w_2)/w5[0,1,1,1,0] + r_2/w5[0,1,1,1,0])
            w_4 = layers_multigrid["CNN3D_prol_2"](w_2) 
            for Jacobi in range(j_itr):
                w_4 = (w_4 - layers_multigrid["CNN3D_A_4"](w_4)/w5[0,1,1,1,0] + r_4/w5[0,1,1,1,0])
            w_8 = layers_multigrid["CNN3D_prol_4"](w_4) 
            for Jacobi in range(j_itr):
                w_8 = (w_8 - layers_multigrid["CNN3D_A_8"](w_8)/w5[0,1,1,1,0] + r_8/w5[0,1,1,1,0])
            w_16 = layers_multigrid["CNN3D_prol_8"](w_8) 
            for Jacobi in range(j_itr):
                w_16 = (w_16 - layers_multigrid["CNN3D_A_16"](w_16)/w5[0,1,1,1,0] + r_16/w5[0,1,1,1,0])
            w_32 = layers_multigrid["CNN3D_prol_16"](w_16) 
#                for Jacobi in range(j_itr):
#                    w_32 = (w_32 - layers_multigrid["CNN3D_A_32"](w_32)/w5[0,1,1,1,0] + r_32/w5[0,1,1,1,0])
#                w_64 = layers_multigrid["CNN3D_prol_32"](w_32) 
            values_p = values_p - w_32
            values_p = MakeTensorZeroInTheMiddle(nx, values_p.shape)(values_p) #[0,:,:,nx-1,0].assign(tf.Variable(tf.zeros((1,nx,nx)))[0,:])         
            values_p = (values_p - layers_multigrid["CNN3D_A_32"](values_p)/w5[0,1,1,1,0] + b/w5[0,1,1,1,0])          
    # correct
    values_p = BoundaryConditionPressure(nx, input_shape)(values_p)
    values_u = values_u - CNN3D_pu(values_p)
    values_v = values_v - CNN3D_pv(values_p)  
    values_w = values_w - CNN3D_pw(values_p)       
    values_u,values_v,values_w = BoundaryConditionVelocity(nx, ub, input_shape)(values_u,values_w,values_v)
    values_u,values_v,values_w = BluffBody(sigma,dt,11,21,27,37,27,37,input_shape)(values_u,values_v,values_w)

    return keras.Model(inputs=[values_u_inp,values_v_inp,values_w_inp,values_p_inp], outputs=[values_u,values_v,values_w,values_p])


@tf.function(jit_compile=True)
def inference_loop(iterator, outfeed_queue, model, steps_per_exe):
    inp = next(iterator)
    for _ in tf.range(steps_per_exe):
        out = model(inp)
        inp["input_1"] = out[0]
        inp["input_2"] = out[1]
        inp["input_3"] = out[2]
        inp["input_4"] = out[3]

        outfeed_queue.enqueue(out)


# # CFD Parameters
# I haven't implemented the Restart == True part of this function yet
# @tf.function(jit_compile=True)
# def AI_HFM():

#     # # Initialise
#     input_shape = (1,nx,ny,nz,1)
#     values_u = tf.zeros(input_shape)
#     values_v = tf.zeros(input_shape)
#     values_w = tf.zeros(input_shape)
#     values_p = tf.zeros(input_shape)

#     multi_itr = 10       # Iterations of multi-grid 
#     j_itr = 1             # Iterations of Jacobi 
#     ntime = 1001      # Time steps
#     n_out = 500           # Results output
#     nrestart = 0          # Last time step for restart
#     ctime_old = 0 
#     mgsolver = True
#     save_fig = False
#     Restart = False
#     ctime = 0         
#     print('Mesh resolution:', values_v.shape)
#     print('Time step:', ntime)
#     print('Initial time:', ctime)

#     if Restart == True:
#         temp1 = np.load('result_3d_27point/u2500.npy').astype('float32')
#         temp2 = np.load('result_3d_27point/v2500.npy').astype('float32')
#         temp3 = np.load('result_3d_27point/w2500.npy').astype('float32')
#         temp4 = np.load('result_3d_27point/p2500.npy').astype('float32')
#         values_u = tf.Variable(values_u)[0,:,:,:,0].assign(tf.convert_to_tensor(temp1))
#         values_v = tf.Variable(values_v)[0,:,:,:,0].assign(tf.convert_to_tensor(temp2))
#         values_w = tf.Variable(values_w)[0,:,:,:,0].assign(tf.convert_to_tensor(temp3))
#         values_p = tf.Variable(values_p)[0,:,:,:,0].assign(tf.convert_to_tensor(temp4))
#         nrestart = 2500
#         ctime_old = nrestart*dt


    
def create_dataset(nx, ny, nz):
    input_shape = (nx,ny,nz,1)
    values_u = np.zeros(input_shape, dtype=dtype_np)
    values_v = np.zeros(input_shape, dtype=dtype_np)
    values_w = np.zeros(input_shape, dtype=dtype_np)
    values_p = np.zeros(input_shape, dtype=dtype_np)
    

    def generator():
        for _ in range(1000):
            yield {"input_1":values_u,
                   "input_2":values_v,
                   "input_3":values_w,
                   "input_4":values_p}

    output_types = {"input_1":dtype,
                    "input_2":dtype,
                    "input_3":dtype,
                    "input_4":dtype}

    output_shapes = {"input_1":input_shape,
                     "input_2":input_shape,
                     "input_3":input_shape,
                     "input_4":input_shape}

    ds = tf.data.Dataset.from_generator(generator,
                                        output_shapes=output_shapes,
                                        output_types=output_types)
    ds = ds.batch(1, drop_remainder=True).repeat()

    return ds #[values_u, values_v, values_w, values_p]
    
if __name__ == "__main__":

    strategy=ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        print(tf.keras.backend.floatx())

        nx = 32
        ny = 32
        nz = 32

        model = create_model(nx, ny, nz)
        model.summary()

        ds = create_dataset(nx, ny, nz)

        # Create IPU infeed/outfeed objects to efficiently send data to/from IPU
        iterator = iter(ds)
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

        # Number of steps to loop over on-IPU per call to strategy.run()
        steps_per_exe = 1000

        # Warm-up, this includes compilation time
        strategy.run(inference_loop, args=[iterator, outfeed_queue, model, steps_per_exe])
        out = outfeed_queue.dequeue()

        # Timed run without compilation
        start = time.time()
        strategy.run(inference_loop, args=[iterator, outfeed_queue, model, steps_per_exe])
        elapsed_time = time.time() - start
        print("Iterated over {} steps in {} seconds".format(steps_per_exe, elapsed_time))

        # Check the output is changing over time, to ensure it's actually reusing the output
        for i in range(0, 1000, 100):
            print(out[0][i,0,0,0,0,0])