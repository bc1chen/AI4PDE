#!/usr/bin/env python

#  Copyright (C) 2023
#  
#  Boyang Chen
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#
#  boyang.chen16@imperial.ac.uk
#  
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation,
#  version 3.0 of the License.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.

#-- Import general libraries
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' ## enable xla devices # Comment out this line if runing on GPU cluster
import numpy as np 
import time 
import math
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import tensorflow as tf

# Import local functions
import AI4CFD_activation_function as f 
import AI4CFD_filters as CNN3D

# Check if GPU is available 
tf.config.list_physical_devices()
is_gpu = len(tf.config.list_physical_devices('GPU')) > 0 
print(is_gpu)

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
dt = 0.0005
dx = 0.01
dy = 0.01
dz = 0.01
Re = 1
ub = 1
nx = 128 #512Test a smaller case for parallelisation with Xiaohu 
ny = 128 #512Test a smaller case for parallelisation with Xiaohu 
nz = 128 #512Test a smaller case for parallelisation with Xiaohu 
nlevel = int(math.log(nz, 2)) + 1 
print('How many levels in multigrid', nlevel)
#################### Create field #####£###############
input_shape = (1,nz, ny,nx,1)
values_u = tf.zeros(input_shape)
values_v = tf.zeros(input_shape)
values_w = tf.zeros(input_shape)
values_ph = tf.zeros(input_shape)
values_pd = tf.zeros(input_shape)
#######################################################
################# Numerical parameters ################
mg_itr_h = 5000             # Iterations of multi-grid 
mg_itr_d = 1000             # Iterations of multi-grid 
j_itr = 1                   # Iterations of Jacobi 
ntime = 20000                   # Time steps
n_out = 400                 # Results output
nrestart = 0                # Last time step for restart
ctime_old = 0               # Last ctime for restart
mgsolver_h = True            # Multigrid solver for hydrostatic pressure
mgsolver_d = True            # Multigrid solver for non-hydrostatic pressure 
LSCALAR = True              # Scalar transport 
LMTI = True                 # Non density for multiphase flows
LIBM = False                # Immersed boundary method 
nsafe = 0.5                 # Continuty equation residuals
ctime = 0                   # Initialise ctime   
save_fig = False            # Save results
Restart = False             # Restart
eplsion_k = 1e-04
################# Physical parameters #################
rho_l = 1000               # Density of liquid phase 
rho_g = 1.0              # Density of gas phase 
g_x = 0;g_y = 0;g_z = -10   # Gravity acceleration (m/s2) 
#######################################################
print('============== Numerical parameters ===============')
print('Mesh resolution:', values_v.shape)
print('Time step:', ntime)
print('Initial time:', ctime)
#######################################################
################# Only for restart ####################
if Restart == True:
    nrestart = 8000
    ctime_old = nrestart*dt
    print('Restart solver!')
#######################################################    
################# Only for scalar #####################
if LSCALAR == True and Restart == False:
    alpha = np.zeros(input_shape).astype('float32')
    alpha[0,0:128,0:100,0:100,0] = 1.0
    alpha = tf.convert_to_tensor(alpha)
    print('Switch on scalar filed solver!')
#######################################################
################# Only for scalar #####################
if LMTI == True and Restart == False:
    rho = tf.zeros(input_shape)
    rho = alpha*rho_l + (1-alpha)*rho_g*50
    print('Solving multiphase flows!')
else:
    print('Solving single-phase flows!')
################# Only for IBM ########################
if LIBM == True:
    mesh = np.load('mesh_512_sk.npy')
    sigma = np.zeros(input_shape).astype('float32')
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            for k in range(1,nz-1):
                if mesh[0][i+64][j+64][k][0] == 0:
                    sigma[0][k][j][i][0] = 100000  
    sigma = tf.convert_to_tensor(sigma)
    print('Switch on IBM solver!')
    print('===================================================')
    plt.imshow(sigma[0,1,:,:,0], cmap='jet')
    plt.colorbar()
    plt.title('South Kensington area')
#######################################################
# # # ################################### # # #
# # # #########   AI4CFD MAIN ########### # # #
# # # ################################### # # #
start = time.time()
for itime in range(1,ntime+1): 
    ctime = ctime + dt  
    temp1 = (CNN3D.zadv(f.boundary_condition_density(rho,nx,ny,nz)*int(abs(g_z))))
    if mgsolver_h == True:
        for multi_grid in range(10):
            w_2 = tf.zeros([1,2,2,2,1])
            # r_512 = CNN3D.A_512(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) - temp1
            r_128 = CNN3D.A_512(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) - temp1          
            # r_256 = CNN3D.res_512(r_512)     
            # r_128 = CNN3D.res_256(r_256)     
            r_64 = CNN3D.res_128(r_128) 
            r_32 = CNN3D.res_64(r_64)         
            r_16 = CNN3D.res_32(r_32) 
            r_8 = CNN3D.res_16(r_16) 
            r_4 = CNN3D.res_8(r_8) 
            r_2 = CNN3D.res_4(r_4) 

            w_2 = w_2 - CNN3D.A_2(w_2)/CNN3D.wA[0,1,1,1,0] + r_2/CNN3D.wA[0,1,1,1,0]
            w_4 = CNN3D.prol_2(w_2) 
            w_4 = w_4 - CNN3D.A_4(w_4)/CNN3D.wA[0,1,1,1,0] + r_4/CNN3D.wA[0,1,1,1,0]       
            w_8 = CNN3D.prol_4(w_4) 
            w_8 = w_8 - CNN3D.A_8(w_8)/CNN3D.wA[0,1,1,1,0] + r_8/CNN3D.wA[0,1,1,1,0]      
            w_16 = CNN3D.prol_8(w_8)
            w_16 = w_16 - CNN3D.A_16(w_16)/CNN3D.wA[0,1,1,1,0] + r_16/CNN3D.wA[0,1,1,1,0]       
            w_32 = CNN3D.prol_16(w_16)   
            w_32 = w_32 - CNN3D.A_32(w_32)/CNN3D.wA[0,1,1,1,0] + r_32/CNN3D.wA[0,1,1,1,0]       
            w_64 = CNN3D.prol_32(w_32)           
            w_64 = w_64 - CNN3D.A_64(w_64)/CNN3D.wA[0,1,1,1,0] + r_64/CNN3D.wA[0,1,1,1,0]       
            w_128 = CNN3D.prol_64(w_64)    
            # w_128 = w_128 - CNN3D.A_128(w_128)/CNN3D.wA[0,1,1,1,0] + r_128/CNN3D.wA[0,1,1,1,0]        
            # w_256 = CNN3D.prol_128(w_128)     
            # w_256 = w_256 - CNN3D.A_256(w_256)/CNN3D.wA[0,1,1,1,0] + r_256/CNN3D.wA[0,1,1,1,0]        
            # w_512 = CNN3D.prol_256(w_256) 

            values_ph = values_ph - w_128
            values_ph = values_ph - CNN3D.A_128(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz))/CNN3D.wA[0,1,1,1,0] + temp1/CNN3D.wA[0,1,1,1,0]
# Momentum equation (two-stepping scheme)
# First step for solving u
    temp1 = f.PG_vector_SAME(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[0],
                           f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[1],
                           f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[2],
                           values_u,values_v,values_w,eplsion_k,rho,nx,ny,nz,dx,dy,dz,dt)[0] * dt / rho - \
    values_u * CNN3D.xadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[0]) * dt - \
    values_v * CNN3D.yadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[0]) * dt - \
    values_w * CNN3D.zadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[0]) * dt 
    temp1 = 0.5 * temp1 + values_u
# First step for solving v
    temp2 = f.PG_vector_SAME(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[0],
                           f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[1],
                           f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[2],
                           values_u,values_v,values_w,eplsion_k,rho,nx,ny,nz,dx,dy,dz,dt)[1] * dt / rho - \
    values_u * CNN3D.xadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[1]) * dt - \
    values_v * CNN3D.yadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[1]) * dt - \
    values_w * CNN3D.zadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[1]) * dt  
    temp2 = 0.5 * temp2 + values_v 
# First step for solving w
    temp3 = f.PG_vector_SAME(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[0],
                           f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[1],
                           f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[2],
                           values_u,values_v,values_w,eplsion_k,rho,nx,ny,nz,dx,dy,dz,dt)[2] * dt / rho - \
    values_u * CNN3D.xadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[2]) * dt - \
    values_v * CNN3D.yadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[2]) * dt - \
    values_w * CNN3D.zadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[2]) * dt  
    temp3 = 0.5 * temp3 + values_w + g_z*dt
# Pressure gradient correction
    temp1 = temp1 - CNN3D.xadv(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) / rho * dt  
    temp2 = temp2 - CNN3D.yadv(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) / rho * dt     
    temp3 = temp3 - CNN3D.zadv(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) / rho * dt 
# Second step for solving u   
    temp4 = f.PG_vector_SAME(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[0],
                           f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[1],
                           f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[2],
                           temp1,temp2,temp3,eplsion_k,rho,nx,ny,nz,dx,dy,dz,dt)[0] * dt / rho - \
    temp1 * CNN3D.xadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[0]) * dt - \
    temp2 * CNN3D.yadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[0]) * dt - \
    temp3 * CNN3D.zadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[0]) * dt 
    values_u = values_u + temp4  
# Second step for solving v   
    temp5 = f.PG_vector_SAME(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[0],
                           f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[1],
                           f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[2],
                           temp1,temp2,temp3,eplsion_k,rho,nx,ny,nz,dx,dy,dz,dt)[1] * dt / rho - \
    temp1 * CNN3D.xadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[1]) * dt - \
    temp2 * CNN3D.yadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[1]) * dt - \
    temp3 * CNN3D.zadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[1]) * dt 
    values_v = values_v + temp5 
# Second step for solving w   
    temp6 = f.PG_vector_SAME(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[0],
                           f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[1],
                           f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[2],
                           temp1,temp2,temp3,eplsion_k,rho,nx,ny,nz,dx,dy,dz,dt)[2] * dt / rho - \
    temp1 * CNN3D.xadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[2]) * dt - \
    temp2 * CNN3D.yadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[2]) * dt - \
    temp3 * CNN3D.zadv(f.boundary_condition_velocity(temp1,temp2,temp3,nx,ny,nz)[2]) * dt 
    values_w = values_w + temp6 + g_z * dt
# grap p (hydrostatic and non-hydrostatic pressure)   
    values_u = values_u - CNN3D.xadv(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) / rho * dt  
    values_v = values_v - CNN3D.yadv(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) / rho * dt     
    values_w = values_w - CNN3D.zadv(f.boundary_condition_pressure_h(values_ph,nx,ny,nz,f.boundary_condition_density(rho,nx,ny,nz),dz)) / rho * dt 
# Transport indicator field 
    temp1 = f.PG_turb_scalar_SAME(f.boundary_condition_indicator_SAME(alpha,nx,ny,nz),
        alpha,values_u,values_v,values_w,eplsion_k,nx,ny,nz,dx,dt) * dt - \
    values_u * CNN3D.xadv(f.boundary_condition_indicator_SAME(alpha,nx,ny,nz)) * dt - \
    values_v * CNN3D.yadv(f.boundary_condition_indicator_SAME(alpha,nx,ny,nz)) * dt - \
    values_w * CNN3D.zadv(f.boundary_condition_indicator_SAME(alpha,nx,ny,nz)) * dt 
    temp1 = alpha + temp1*0.5
    
    temp2 = f.PG_turb_scalar_SAME(tf.math.maximum(tf.math.minimum(f.boundary_condition_indicator_SAME(temp1,nx,ny,nz),1),0),
        tf.math.maximum(tf.math.minimum(temp1,1),0),values_u,values_v,values_w,eplsion_k,nx,ny,nz,dx,dt) * dt - \
    values_u * CNN3D.xadv(tf.math.maximum(tf.math.minimum(f.boundary_condition_indicator_SAME(temp1,nx,ny,nz),1),0)) * dt - \
    values_v * CNN3D.yadv(tf.math.maximum(tf.math.minimum(f.boundary_condition_indicator_SAME(temp1,nx,ny,nz),1),0)) * dt - \
    values_w * CNN3D.zadv(tf.math.maximum(tf.math.minimum(f.boundary_condition_indicator_SAME(temp1,nx,ny,nz),1),0)) * dt 
    alpha = alpha + temp2
# Avoid sharp interfacing    
    alpha = tf.math.minimum(alpha,1)
    alpha = tf.math.maximum(alpha,0)
    temp6 = rho
    rho = alpha*rho_l + (1 - alpha) * rho_g * 50  
# Multigrid for non-hydrostatic pressure
    temp1 = -(-CNN3D.xadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[0] *
        f.boundary_condition_density(rho,nx,ny,nz)) + \
    -CNN3D.yadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[1] *
        f.boundary_condition_density(rho,nx,ny,nz)) + \
    -CNN3D.zadv(f.boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz)[2] *
        f.boundary_condition_density(rho,nx,ny,nz)) - (rho - temp6) / dt) / dt
    if mgsolver_d == True:
        for multi_grid in range(10):
            w_2 = tf.zeros([1,2,2,2,1])
            # r_512 = CNN3D.A_512(f.boundary_condition_pressure_d(values_pd,nx,ny,nz)) - temp1
            r_128 = CNN3D.A_128(f.boundary_condition_pressure_d(values_pd,nx,ny,nz)) - temp1
            # r_256 = CNN3D.res_512(r_512)     
            # r_128 = CNN3D.res_256(r_256)     
            r_64 = CNN3D.res_128(r_128) 
            r_32 = CNN3D.res_64(r_64)         
            r_16 = CNN3D.res_32(r_32) 
            r_8 = CNN3D.res_16(r_16) 
            r_4 = CNN3D.res_8(r_8) 
            r_2 = CNN3D.res_4(r_4) 

            w_2 = w_2 - CNN3D.A_2(w_2)/CNN3D.wA[0,1,1,1,0] + r_2/CNN3D.wA[0,1,1,1,0]
            w_4 = CNN3D.prol_2(w_2) 
            w_4 = w_4 - CNN3D.A_4(w_4)/CNN3D.wA[0,1,1,1,0] + r_4/CNN3D.wA[0,1,1,1,0]      
            w_8 = CNN3D.prol_4(w_4) 
            w_8 = w_8 - CNN3D.A_8(w_8)/CNN3D.wA[0,1,1,1,0] + r_8/CNN3D.wA[0,1,1,1,0]     
            w_16 = CNN3D.prol_8(w_8)
            w_16 = w_16 - CNN3D.A_16(w_16)/CNN3D.wA[0,1,1,1,0] + r_16/CNN3D.wA[0,1,1,1,0]      
            w_32 = CNN3D.prol_16(w_16)   
            w_32 = w_32 - CNN3D.A_32(w_32)/CNN3D.wA[0,1,1,1,0] + r_32/CNN3D.wA[0,1,1,1,0]   
            w_64 = CNN3D.prol_32(w_32)           
            w_64 = w_64 - CNN3D.A_64(w_64)/CNN3D.wA[0,1,1,1,0] + r_64/CNN3D.wA[0,1,1,1,0]     
            w_128 = CNN3D.prol_64(w_64)    
            # w_128 = w_128 - CNN3D.A_128(w_128)/CNN3D.wA[0,1,1,1,0] + r_128/CNN3D.wA[0,1,1,1,0]        
            # w_256 = CNN3D.prol_128(w_128)     
            # w_256 = w_256 - CNN3D.A_256(w_256)/CNN3D.wA[0,1,1,1,0] + r_256/CNN3D.wA[0,1,1,1,0]        
            # w_512 = CNN3D.prol_256(w_256) 

            values_pd = values_pd - w_128
            values_pd = values_pd - CNN3D.A_128(f.boundary_condition_pressure_d(values_pd,nx,ny,nz))/CNN3D.wA[0,1,1,1,0] + temp1/CNN3D.wA[0,1,1,1,0]
# grap p (hydrostatic and non-hydrostatic pressure)   
    values_u = values_u + CNN3D.xadv(f.boundary_condition_pressure_d(values_pd,nx,ny,nz))/rho*dt 
    values_v = values_v + CNN3D.yadv(f.boundary_condition_pressure_d(values_pd,nx,ny,nz))/rho*dt 
    values_w = values_w + CNN3D.zadv(f.boundary_condition_pressure_d(values_pd,nx,ny,nz))/rho*dt   
# output   
    print('Time step:', itime) 
    print('Pressure error:', np.max(w_128), 'cty equation residual:', np.max(r_128))
    print('========================================================')
    if np.max(np.abs(w_128)) > 80000.0:
        print('Not converged !!!!!!')
        break
    if save_fig == True:
        f.save_data(alpha,rho,n_out,itime+nrestart)
end = time.time()
print('time',(end-start))
