#!/usr/bin/env python3

#-- Import general libraries
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import math
import numpy as np

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
dx = 0.01
dy = 0.01
dz = 0.01
nx = 512
ny = 512
nz = 512
nlevel = int(math.log(nz, 2)) + 1 
bias_initializer = tf.keras.initializers.constant(np.zeros((1,)))
# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
# Laplacian filters 
pd1 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
pd2 = [[3/26, 6/26,  3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26,  3/26]]
pd3 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
w1 = np.zeros([1,3,3,3,1])
wA = np.zeros([1,3,3,3,1])
w1[0,0,:,:,0] = np.array(pd1)/dx**2
w1[0,1,:,:,0] = np.array(pd2)/dx**2
w1[0,2,:,:,0] = np.array(pd3)/dx**2
wA[0,0,:,:,0] = -np.array(pd1)/dx**2
wA[0,1,:,:,0] = -np.array(pd2)/dx**2
wA[0,2,:,:,0] = -np.array(pd3)/dx**2
# Gradient filters
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
w2 = np.zeros([1,3,3,3,1])
w3 = np.zeros([1,3,3,3,1])
w4 = np.zeros([1,3,3,3,1])
w2[0,0,:,:,0] = -np.array(p_div_x1)/dx
w2[0,1,:,:,0] = -np.array(p_div_x2)/dx
w2[0,2,:,:,0] = -np.array(p_div_x3)/dx
w3[0,0,:,:,0] = -np.array(p_div_y1)/dx
w3[0,1,:,:,0] = -np.array(p_div_y2)/dx
w3[0,2,:,:,0] = -np.array(p_div_y3)/dx
w4[0,0,:,:,0] = -np.array(p_div_z1)/dx 
w4[0,1,:,:,0] = -np.array(p_div_z2)/dx
w4[0,2,:,:,0] = -np.array(p_div_z3)/dx
# Curvature Laplacian filters
curvature_x1 = [[-0.1875, 0.375,  -0.1875],
       [-0.75, 1.5,  -0.75],
       [-0.1875, 0.375,  -0.1875]]
curvature_x2= [[-0.75, 1.5,  -0.75],
       [-3.0, 6.0,  -3.0],
       [-0.75, 1.5,  -0.75]]
curvature_x3 = [[-0.1875, 0.375,  -0.1875],
       [-0.75, 1.5,  -0.75],
       [-0.1875, 0.375,  -0.1875]]
curvature_y1 = [[-0.1875, -0.75,  -0.1875],
       [0.375, 1.5,  0.375],
       [-0.1875, -0.75,  -0.1875]]
curvature_y2= [[-0.75, -3.0,  -0.75],
       [1.5, 6.0,  1.5],
       [-0.75, -3.0,  -0.75]]
curvature_y3 = [[-0.1875, -0.75,  -0.1875],
       [0.375, 1.5,  0.375],
       [-0.1875, -0.75,  -0.1875]]
curvature_z1 = [[-0.1875, -0.75,  -0.1875],
       [-0.75, -3.0,  -0.75],
       [-0.1875, -0.75,  -0.1875]]
curvature_z2= [[0.375, 1.5,  0.375],
       [1.5, 6.0,  1.5],
       [0.375, 1.5,  0.375]]
curvature_z3 = [[-0.1875, -0.75,  -0.1875],
       [-0.75, -3.0,  -0.75],
       [-0.1875, -0.75,  -0.1875]]
AD2_x = np.zeros([1,3,3,3,1])
AD2_y = np.zeros([1,3,3,3,1])
AD2_z = np.zeros([1,3,3,3,1])
AD2_x[0,0,:,:,0] = -np.array(curvature_x1)/dx**2
AD2_x[0,1,:,:,0] = -np.array(curvature_x2)/dx**2
AD2_x[0,2,:,:,0] = -np.array(curvature_x3)/dx**2
AD2_y[0,0,:,:,0] = -np.array(curvature_y1)/dx**2
AD2_y[0,1,:,:,0] = -np.array(curvature_y2)/dx**2
AD2_y[0,2,:,:,0] = -np.array(curvature_y3)/dx**2
AD2_z[0,0,:,:,0] = -np.array(curvature_z1)/dx**2
AD2_z[0,1,:,:,0] = -np.array(curvature_z2)/dx**2
AD2_z[0,2,:,:,0] = -np.array(curvature_z3)/dx**2
# Restriction filters
w_res = np.zeros([1,2,2,2,1])
w_res[0,:,:,:,0] = 0.125
# Detecting filters
wxu = np.zeros([1,3,3,3,1])
wxd = np.zeros([1,3,3,3,1])
wyu = np.zeros([1,3,3,3,1])
wyd = np.zeros([1,3,3,3,1])
wzu = np.zeros([1,3,3,3,1])
wzd = np.zeros([1,3,3,3,1])
wxu[0][1][1][1][0] = 1.0
wxu[0][1][1][0][0] = -1.0
wxd[0][1][1][1][0] = -1.0
wxd[0][1][1][2][0] = 1.0
wyu[0][1][1][1][0] = 1.0
wyu[0][1][0][1][0] = -1.0
wyd[0][1][1][1][0] = -1.0
wyd[0][1][2][1][0] = 1.0
wzu[0][1][1][1][0] = 1.0
wzu[0][0][1][1][0] = -1.0
wzd[0][1][1][1][0] = -1.0
wzd[0][2][1][1][0] = 1.0
# Initialise the CNNs filters 
kernel_initializer_1 = tf.keras.initializers.constant(w1)
kernel_initializer_2 = tf.keras.initializers.constant(w2)
kernel_initializer_3 = tf.keras.initializers.constant(w3)
kernel_initializer_4 = tf.keras.initializers.constant(w4)
kernel_initializer_A = tf.keras.initializers.constant(wA)
kernel_initializer_AD2_x = tf.keras.initializers.constant(AD2_x)
kernel_initializer_AD2_y = tf.keras.initializers.constant(AD2_y)
kernel_initializer_AD2_z = tf.keras.initializers.constant(AD2_z)
kernel_initializer_w_res = tf.keras.initializers.constant(w_res)
kernel_initializer_wxu = tf.keras.initializers.constant(wxu)
kernel_initializer_wxd = tf.keras.initializers.constant(wxd)
kernel_initializer_wyu = tf.keras.initializers.constant(wyu)
kernel_initializer_wyd = tf.keras.initializers.constant(wyd)
kernel_initializer_wzu = tf.keras.initializers.constant(wzu)
kernel_initializer_wzd = tf.keras.initializers.constant(wzd)
# Initialise the CNNs filters 
difx = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_AD2_x,
                                bias_initializer=bias_initializer),
])

dify = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_AD2_y,
                                bias_initializer=bias_initializer),
])

difz = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_AD2_z,
                                bias_initializer=bias_initializer),
])
dif = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])

xadv = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

yadv = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_3,
                                bias_initializer=bias_initializer),
])

zadv = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_4,
                                bias_initializer=bias_initializer),
])
for i in range(nlevel):
       if i < nlevel-1:
              locals()['A_'+str(2**i)] = keras.models.Sequential([
                     keras.layers.InputLayer(input_shape=(int(nz*0.5**(nlevel-1-i)), int(ny*0.5**(nlevel-1-i)), int(nx*0.5**(nlevel-1-i)), 1)),
                     tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer=kernel_initializer_A,
                            bias_initializer=bias_initializer)
              ])
       else:
              locals()['A_'+str(2**i)] = keras.models.Sequential([
                     keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
                     tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                            kernel_initializer=kernel_initializer_A,
                            bias_initializer=bias_initializer)
              ]) 
for i in range(nlevel-1):
    locals()['res_'+str(2**(i+1))] = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(int(nz*0.5**(nlevel-2-i)), int(ny*0.5**(nlevel-2-i)), int(nx*0.5**(nlevel-2-i)), 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_w_res,
                                bias_initializer=bias_initializer),   
    ])    
for i in range(nlevel-1):
    locals()['prol_'+str(2**i)] = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(1*2**i, 1*1*2**i, 1*1*2**i, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
    ])
wxu = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_wxu,
                                bias_initializer=bias_initializer),
])
wxd = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_wxd,
                                bias_initializer=bias_initializer),
])
wyu = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_wyu,
                                bias_initializer=bias_initializer),
])
wyd = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_wyd,
                                bias_initializer=bias_initializer),
])
wzd = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_wzd,
                                bias_initializer=bias_initializer),
])
wzu = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nz+2, ny+2, nx+2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_wzu,
                                bias_initializer=bias_initializer),
])
