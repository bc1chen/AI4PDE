# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
from time import perf_counter

# import halo_exchanged
from halo_exchange_upgraded import HaloExchange
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid dnn issue

# Parameters to be defined for different grid size and conductivity
dt = 10      # Time step (s)
dx = 1       # Grid size in x      
Dx = 0.1   # Conductivity in x    
# Parameters for the computational domain
alpha = 1    # relaxation coefficient for Jacobi iteration (from 0 to 1)
nx = 128     # Grid point in x
ub = 1       # Velocity (1m/s)

# the weights matrix
w1 = np.zeros([1,2,1])
w2 = np.zeros([1,3,1])
w1[0,:,0] = 0.5
w2[0][0][0] = - ub*dt/(dx) - Dx*dt/dx**2
w2[0][1][0] = 1 + 2*Dx*dt/dx**2 + ub*dt/(dx)
w2[0][2][0] =  - Dx*dt/dx**2

kernel_initializer_1 = tf.keras.initializers.constant(w1)
kernel_initializer_2 = tf.keras.initializers.constant(w2)
bias_initializer = tf.keras.initializers.constant(np.zeros((1,)))

CNN3D_A_128 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nx, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer)
])

CNN3D_A_66 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(66, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_34 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(34, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_18 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(18, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_10 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(10, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_6 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(6, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_1 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(1, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

# restrictions

CNN3D_res_128 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nx, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),   
])    
CNN3D_res_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),  
])
CNN3D_res_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer), 
])
CNN3D_res_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer), 
])
CNN3D_res_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])
CNN3D_res_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])
CNN3D_res_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 1)),
         tf.keras.layers.Conv1D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])

# prolongation
CNN3D_prol_1 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(1, 1)),
         tf.keras.layers.UpSampling1D(size=(2)),
])

CNN3D_prol_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 1)),
         tf.keras.layers.UpSampling1D(size=(2)),
])

CNN3D_prol_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 1)),
         tf.keras.layers.UpSampling1D(size=(2)),
])

CNN3D_prol_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 1)),
         tf.keras.layers.UpSampling1D(size=(2)),   
])

CNN3D_prol_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 1)),
         tf.keras.layers.UpSampling1D(size=(2)), 
])

CNN3D_prol_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 1)),
         tf.keras.layers.UpSampling1D(size=(2)),   
])

CNN3D_prol_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 1)),
         tf.keras.layers.UpSampling1D(size=(2)),
])

# you might need incorporate with SFC code here
T = np.zeros([nx])
for i in range(40):
    T[i+43] = 1 

he = HaloExchange(structured=True,tensor_used=True,double_precision=True,corner_exchanged=False)
sub_nx, current_domain = he.initialization(T,is_periodic=False,is_reordered=False)
sub_x = sub_nx + 2
rank = he.rank

# need 1 update before the multigrid
current_domain = he.structured_halo_update_1D(current_domain)
current_domain = current_domain.numpy().reshape(sub_x,)

# convert current domain to tensor
input_shape = (1,sub_x,1)
values = tf.zeros(input_shape) # (1,66,1)
values = tf.Variable(values)[0,:,0].assign(tf.Variable(values)[0,:,0]+tf.convert_to_tensor(current_domain.astype('float32')))

start = perf_counter()
b = tf.reshape(values[0,1:-1,0],(1,sub_nx,1))           # only for one time step
# b = values         # only for one time step
multi_itr = 1000      # multigrid iteration
j_itr = 1          # jacobi iteration 

for multi_grid in range(multi_itr):    
    w = np.zeros([1,1,1]) # from 1 -> 64
# --------- Calculate Residual based on initial guess --------  
    # r = CNN3D_A_128(values) - b 
    r = CNN3D_A_66(values) - b
# ------------------------------------------------------------  

# --------- Interpolate Residual from finer to coaser mesh --------  
    r_32 = CNN3D_res_64(r)
    r_16 = CNN3D_res_32(r_32)
    r_8 = CNN3D_res_16(r_16) 
    r_4 = CNN3D_res_8(r_8) 
    r_2 = CNN3D_res_4(r_4) 
    r_1 = CNN3D_res_2(r_2)    
    
# -----------------------------------------------------------------      

# --------- Interpolate Residual from coaser to finer mesh --------  
    for Jacobi in range(j_itr):
        w = w - CNN3D_A_1(w)/w2[0][1][0] + r_1/w2[0][1][0]
    w = w - CNN3D_A_1(w)/w2[0][1][0] + r_1/w2[0][1][0]
    
    # print('RESIDUAL 1 SHAPE: ', w.shape)

    w_2 = CNN3D_prol_1(w)
    w_t1 = he.padding_block_halo_1D(w_2,1)
    w_t1 = he.structured_halo_update_1D(w_t1)     
    for Jacobi in range(j_itr):
        temp = CNN3D_A_4(w_t1)
        w_2 = w_2 - temp/w2[0][1][0] + r_2/w2[0][1][0]
        
    # print('RESIDUAL 2 SHAPE: ', w_2.shape)

    w_4 = CNN3D_prol_2(w_2)
    w_t2 = he.padding_block_halo_1D(w_4,1)
    w_t2 = he.structured_halo_update_1D(w_t2)   
    for Jacobi in range(j_itr):
        temp = CNN3D_A_6(w_t2)
        w_4 = w_4 - temp/w2[0][1][0] + r_4/w2[0][1][0]
        
    # print('RESIDUAL 4 SHAPE: ', w_4.shape)

    w_8 = CNN3D_prol_4(w_4)
    w_t3 = he.padding_block_halo_1D(w_8,1)
    w_t3 = he.structured_halo_update_1D(w_t3)    
    for Jacobi in range(j_itr):
        temp = CNN3D_A_10(w_t3)
        w_8 = w_8 - temp/w2[0][1][0] + r_8/w2[0][1][0]
        
    # print('RESIDUAL 8 SHAPE: ', w_8.shape)

    w_16 = CNN3D_prol_8(w_8)
    w_t4 = he.padding_block_halo_1D(w_16,1)
    w_t4 = he.structured_halo_update_1D(w_t4)  
    for Jacobi in range(j_itr):
        temp = CNN3D_A_18(w_t4)
        w_16 = w_16 - temp/w2[0][1][0] + r_16/w2[0][1][0]
        
    # print('RESIDUAL 16 SHAPE: ', w_16.shape)
    
    w_32 = CNN3D_prol_16(w_16)
    w_t5 = he.padding_block_halo_1D(w_32,1)
    w_t5 = he.structured_halo_update_1D(w_t5)  
    for Jacobi in range(j_itr):
        temp = CNN3D_A_34(w_t5)
        w_32 = w_32 - temp/w2[0][1][0] + r_32/w2[0][1][0]

    # print('RESIDUAL 32 SHAPE: ', w_32.shape)

    w_64 = CNN3D_prol_32(w_32)
    w_t6 = he.padding_block_halo_1D(w_64,1)
    w_t6 = he.structured_halo_update_1D(w_t6)  
    for Jacobi in range(j_itr):
        temp = CNN3D_A_66(w_t6)
        w_64 = w_64 - temp/w2[0][1][0] + r/w2[0][1][0]
        
    # print('RESIDUAL 64 SHAPE: ', w_64.shape)

    #w_128 = CNN3D_prol_64(w_64)
    #w_128 = w_128 - CNN3D_A_128(w_128)/w2[0][1][0] + r/w2[0][1][0]
# ----------------------------------------------------------------- 

# --------- Correct initial guess --------  
    #values = values - w_128 
    #values = values - CNN3D_A_128(values)/w2[0][1][0] + b/w2[0][1][0]

    w_64 = he.padding_block_halo_1D(w_64,1)
    w_64 = he.structured_halo_update_1D(w_64)
    
    values = values - w_64
    tempVal = tf.reshape(values[0,1:-1,0],(1,64,1))
    tempVal = tempVal - CNN3D_A_66(values)/w2[0][1][0] + b/w2[0][1][0]
    values = he.padding_block_halo_1D(tempVal,1)
    values = he.structured_halo_update_1D(values)
    
# ----------------------------------------  
end = perf_counter()
print('Computational time(s):',(end-start))
print('Multigrid iterations:', multi_itr)
print('Jacobi iterations:', j_itr)

np.save("parallel_out/parallel_res_{}".format(rank),values[0,1:-1,0])
np.save("parallel_residuals/w_{}".format(rank),w)
np.save("parallel_residuals/w2_{}".format(rank),w_2)
np.save("parallel_residuals/w4_{}".format(rank),w_4)
np.save("parallel_residuals/w8_{}".format(rank),w_8)
np.save("parallel_residuals/w16_{}".format(rank),w_16)
np.save("parallel_residuals/w32_{}".format(rank),w_32)
np.save("parallel_residuals/w64_{}".format(rank),w_64)
