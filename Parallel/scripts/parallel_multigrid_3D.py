
#============================== Imports ==================================#
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # avoid tf dnn flag issue

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import math
from time import perf_counter

from halo_exchange_upgraded import HaloExchange

#============================== Initialization of original problem ==================================#
dx = 1       # Grid size in x    
dy = 1       # Grid size in y       
dz = 1       # Grid size in z    
Dx = 100.0   # Conductivity in x    
Dy = 100.0   # Conductivity in y
Dz = 100.0    # Conductivity in z
# Parameters for the computational domain
alpha = 1    # relaxation coefficient for Jacobi iteration (from 0 to 1)
nx = 128     # Grid point in x
ny = 128     # Grid point in y
nz = 128     # Grid point in z

# the weights matrix
w1 = np.zeros([1,2,2,2,1])
w2 = np.zeros([1,3,3,3,1])
w1[0,:,:,:,0] = 0.125
pd1 = [[0.0, 0.0,  0.0],
       [0.0, 1.0*Dz/dz**2,  0.0],
       [0.0, 0.0,  0.0]]
pd2 = [[0.0, 1.0*Dy/dy**2,  0.0],
       [1.0*Dx/dx**2,  -(2*Dx/dx**2+2*Dy/dy**2+2*Dz/dz**2),  1.0*Dx/dx**2],
       [0.0, 1.0*Dy/dy**2,  0.0]]
pd3 = [[0.0, 0.0,  0.0],
       [0.0, 1.0*Dz/dz**2,  0.0],
       [0.0, 0.0,  0.0]]
w2[0,0,:,:,0] = -np.array(pd1) 
w2[0,1,:,:,0] = -np.array(pd2) 
w2[0,2,:,:,0] = -np.array(pd3) 

kernel_initializer_1 = tf.keras.initializers.constant(w1)
kernel_initializer_2 = tf.keras.initializers.constant(w2)
bias_initializer = tf.keras.initializers.constant(np.zeros((1,)) )

T = np.zeros([nx,ny,nz]) # problem space (128,128,128)
gamma = 10
x0 = 0 
y0 = 0
z0 = 0
x = np.zeros([1,nx])
y = np.zeros([1,ny])
z = np.zeros([1,nz])

for i in range(40):
    for j in range(40):
        for k in range(40):
            T[i+43][j+43][k+43] = 1 

# T -> (1,128,128,128,1)
he = HaloExchange(structured=True,tensor_used=True,double_precision=True,corner_exchanged=True)
sub_nx, sub_ny, sub_nz,current_domain = he.initialization(T,is_periodic=False,is_reordered=False)

input_shape = (1,sub_nx+2,sub_ny+2,sub_nz+2,1)
values = tf.zeros(input_shape,tf.float32)
rank = he.rank # get the process rank

# update halo once as the start
current_domain = he.structured_halo_update_3D(current_domain)
# print(current_domain.shape)
current_domain = current_domain.numpy().reshape(sub_nx+2, sub_ny+2, sub_nz+2)

#============================== Initialization CNN layers ==================================#
# CNN layers
CNN3D_A_128 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer)
])

CNN3D_A_66 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(66, 66, 66, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 64, 64, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_34 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(34, 34, 34, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 32, 32, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_18 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(18, 18, 18, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 16, 16, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_10 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(10, 10, 10, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 8, 8, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_6 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(6, 6, 6, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 4, 4, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_3 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(3, 3, 3, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 2, 2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_1 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(1, 1, 1, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

#============================== Restriction process ==================================#

CNN3D_res_128 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),   
])    

CNN3D_res_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 64, 64, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),  
])
CNN3D_res_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 32, 32, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer), 
])
CNN3D_res_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 16, 16, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer), 
])
CNN3D_res_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 8, 8, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])
CNN3D_res_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 4, 4, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])

CNN3D_res_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 2, 2, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])


#============================== Prolongation process ==================================#
CNN3D_prol_1 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(1, 1, 1, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 2, 2, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 4, 4, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 8, 8, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),   
])

CNN3D_prol_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 16, 16, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)), 
])

CNN3D_prol_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 32, 32, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),   
])

CNN3D_prol_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 64, 64, 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

#============================== Skipping layer ==================================#
start_time =  perf_counter()

# F-cycle multigrid method
temp1 = tf.Variable(values)
temp1[0,:,:,:,0].assign(temp1[0,:,:,:,0]+tf.convert_to_tensor(current_domain.astype('float32'))) # for parallel we add subdomain
values = temp1

# b = values # only for one time step
b = tf.reshape(values[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))           # only for one time step
multi_itr = 100
j_itr = 1 # Jacobi iteration number


# halo-update at each Jacobi iteration
for multi_grid in range(multi_itr):
    w = np.zeros([1,1,1,1,1])
    r = CNN3D_A_66(values) - b # compute the residual
    
    np.save('parallel_residuals/parallel_multigrid_3d_res_{}'.format(rank),r)
    
    # r_64 = CNN3D_res_128(r) 
    r_32 = CNN3D_res_64(r) # 128
    r_16 = CNN3D_res_32(r_32) # 64
    r_8 = CNN3D_res_16(r_16) # 32
    r_4 = CNN3D_res_8(r_8) # 16
    r_2 = CNN3D_res_4(r_4) # 8
    r_1 = CNN3D_res_2(r_2) # 4

    # I suppose here is the proess of Jacobi smoothing followed by prolongation (and correction)
    for Jacobi in range(j_itr):
        w = w - CNN3D_A_1(w)/w2[0,1,1,1,0] + r_1/w2[0,1,1,1,0]
    w = w - CNN3D_A_1(w)/w2[0,1,1,1,0] + r_1/w2[0,1,1,1,0]

    w_2 = CNN3D_prol_1(w)
    w_t1 = he.padding_block_halo_3D(w_2,1)
    w_t1 = he.structured_halo_update_3D(w_t1)
    for Jacobi in range(j_itr):
      temp = CNN3D_A_4(w_t1)
      w_2 = w_2 - temp/w2[0,1,1,1,0] + r_2/w2[0,1,1,1,0]

    w_4 = CNN3D_prol_2(w_2)
    w_t2 = he.padding_block_halo_3D(w_4,1)
    w_t2 = he.structured_halo_update_3D(w_t2)
    for Jacobi in range(j_itr):
      temp = CNN3D_A_6(w_t2)
      w_4 = w_4 - temp/w2[0,1,1,1,0] + r_4/w2[0,1,1,1,0]

    w_8 = CNN3D_prol_4(w_4)
    w_t3 = he.padding_block_halo_3D(w_8,1)
    w_t3 = he.structured_halo_update_3D(w_t3)
    for Jacobi in range(j_itr):
      temp = CNN3D_A_10(w_t3)
      w_8 = w_8 - temp/w2[0,1,1,1,0] + r_8/w2[0,1,1,1,0]

    w_16 = CNN3D_prol_8(w_8)
    w_t4 = he.padding_block_halo_3D(w_16,1)
    w_t4 = he.structured_halo_update_3D(w_t4)  
    for Jacobi in range(j_itr):
      temp = CNN3D_A_18(w_t4)
      w_16 = w_16 - temp/w2[0,1,1,1,0] + r_16/w2[0,1,1,1,0]

    w_32 = CNN3D_prol_16(w_16) 
    w_t5 = he.padding_block_halo_3D(w_32,1)
    w_t5 = he.structured_halo_update_3D(w_t5)  
    for Jacobi in range(j_itr):
      temp = CNN3D_A_34(w_t5)
      w_32 = w_32 - temp/w2[0,1,1,1,0] + r_32/w2[0,1,1,1,0]

    w_64 = CNN3D_prol_32(w_32) 
    w_t6 = he.padding_block_halo_3D(w_64,1)
    w_t6 = he.structured_halo_update_3D(w_t6)  
    for Jacobi in range(j_itr):
      temp = CNN3D_A_66(w_t6)
      w_64 = w_64 - temp/w2[0,1,1,1,0] + r/w2[0,1,1,1,0]
      
    w_64 = he.padding_block_halo_3D(w_64,1)
    w_64 = he.structured_halo_update_3D(w_64)

    values = values - w_64
    tempVal = tf.reshape(values[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_nz,1))
    tempVal = tempVal - CNN3D_A_66(values)/w2[0,1,1,1,0] + b/w2[0,1,1,1,0]
    values = he.padding_block_halo_3D(tempVal,1)
    values = he.structured_halo_update_3D(values)
    
end_time = perf_counter()

# save final result and the prolongations (residual on different grids)
np.save("parallel_out/parallel_AD_multigrid_3D_result_proc_{}.npy".format(rank),values)
np.save("parallel_out/parallel_AD_multigrid_3D_w_proc_{}.npy".format(rank),w)
np.save("parallel_out/parallel_AD_multigrid_3D_w2_proc_{}.npy".format(rank),w_2)
np.save("parallel_out/parallel_AD_multigrid_3D_w4_proc_{}.npy".format(rank),w_4)
np.save("parallel_out/parallel_AD_multigrid_3D_w8_proc_{}.npy".format(rank),w_8)
np.save("parallel_out/parallel_AD_multigrid_3D_w16_proc_{}.npy".format(rank),w_16)
np.save("parallel_out/parallel_AD_multigrid_3D_w32_proc_{}.npy".format(rank),w_32)
np.save("parallel_out/parallel_AD_multigrid_3D_w64_proc_{}.npy".format(rank),w_64)

print(f"[INFO] Problem solved in {end_time - start_time:0.4f} seconds using parallel multigrid.")
