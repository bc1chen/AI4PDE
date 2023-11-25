#!/usr/bin/env python

#  Copyright (C) 2023
#  
#  Boyang Chen, Christopher Pain
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#
#  boyang.chen16@imperial.ac.uk; c.pain@imperial.ac.uk
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
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import local functions
# import AI4SWE_activation_function_torch as f 
# import AI4SWE_filters_torch as CNN2D

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
dt = 0.5
dx = 5.0
nx = 400 
ny = 400 
CFL = np.sqrt(9.81*4)*dt/dx
print('Grid size:', dx,'CFL:', CFL)
# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0])
# Isotropic Laplacian  
w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
        [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
        [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]])
# Gradient in x 
w2 = torch.tensor([[[[1/(12*dx)], [0.0], [-1/(12*dx)]],
        [[1/(3*dx)] , [0.0], [-1/(3*dx)]] ,
        [[1/(12*dx)], [0.0], [-1/(12*dx)]]]])
# Gradient in y 
w3 = torch.tensor([[[[-1/(12*dx)], [-1/(3*dx)], [-1/(12*dx)]],
        [[0.0]       , [0.0]      , [0.0]]       ,
        [[1/(12*dx)] , [1/(3*dx)] , [1/(12*dx)]]]])
# Consistant mass matrix 
wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
        [[0.11] ,  [0.44], [0.11]],
        [[0.028], [0.11] , [0.028]]]])

wavg = torch.tensor([[[[0.0], [0.0] , [0.0]],
        [[1] ,  [0.0], [0.0]],
        [[0.0], [0.0] , [0.0]]]])

w1 = torch.reshape(w1,(1,1,3,3))
w2 = torch.reshape(w2,(1,1,3,3))
w3 = torch.reshape(w3,(1,1,3,3))
wm = torch.reshape(wm,(1,1,3,3))
wavg = torch.reshape(wavg,(1,1,3,3))
#######################################################
################# Numerical parameters ################
ntime = 36000              # Time steps
n_out = 100                 # Results output
nrestart = 0                # Last time step for restart
ctime_old = 0               # Last ctime for restart
mgsolver = True             # Multigrid solver for non-hydrostatic pressure 
nsafe = 0.5                 # Continuty equation residuals
ctime = 0                   # Initialise ctime   
save_fig = True             # Save results
Restart = False             # Restart
eplsion_k = 1e-03          # Stablisatin factor in Petrov-Galerkin for velocity
eplsion_eta = 1e-03         # Stablisatin factor in Petrov-Galerkin for height
beta = 4                    # diagonal factor in mass term
real_time = 0
istep = 0
tt = 0
################# Physical parameters #################
g_x = 0;g_y = 0;g_z = 9.81  # Gravity acceleration (m/s2) 
rho = 1/g_z                 # Resulting density
diag = -np.array(w1)[0,0,1,1]
#######################################################
#################### Create field (tensor) ####################
# Create field (tensor)
input_shape = (1, 1, ny, nx)
values_h = torch.zeros(input_shape, device=device) # dbuging !!!!!!!!!!!!!
values_H = torch.zeros(input_shape, device=device)
values_u = torch.zeros(input_shape, device=device)
values_v = torch.zeros(input_shape, device=device)
a_u = torch.zeros(input_shape, device=device)
a_v = torch.zeros(input_shape, device=device)
b_u = torch.zeros(input_shape, device=device)
b_v = torch.zeros(input_shape, device=device)
c_u = torch.zeros(input_shape, device=device)
c_v = torch.zeros(input_shape, device=device)
eta1 = torch.zeros(input_shape, device=device)
eta2 = torch.zeros(input_shape, device=device)
values_hh = torch.zeros(input_shape, device=device)
dif_values_h = torch.zeros(input_shape, device=device)
values_h_old = torch.zeros(input_shape, device=device)
sigma_q = torch.zeros(input_shape, device=device)
k_u = torch.zeros(input_shape, device=device)
k_v = torch.zeros(input_shape, device=device)
k_x = torch.zeros(input_shape, device=device)
k_y = torch.zeros(input_shape, device=device)
b = torch.zeros(input_shape, device=device)

# values_u_new = torch.zeros(input_shape, device=device)
# values_v_new = torch.zeros(input_shape, device=device)
# values_h_new = torch.zeros(input_shape, device=device)
# Padding
input_shape_pd = (1, 1, ny + 2, nx + 2)
values_uu = torch.zeros(input_shape_pd, device=device)
values_vv = torch.zeros(input_shape_pd, device=device)
b_uu = torch.zeros(input_shape_pd, device=device)
b_vv = torch.zeros(input_shape_pd, device=device)
eta1_p = torch.zeros(input_shape_pd, device=device)
dif_values_hh = torch.zeros(input_shape_pd, device=device)
values_hhp = torch.zeros(input_shape_pd, device=device)
values_hp = torch.zeros(input_shape_pd, device=device)
k_uu = torch.zeros(input_shape_pd, device=device)
k_vv = torch.zeros(input_shape_pd, device=device)
# stablisation factor
k1 = torch.ones(input_shape, device=device)*eplsion_eta
k2 = torch.zeros(input_shape, device=device)
k3 = torch.ones(input_shape, device=device)*dx**2*0.25/dt
#######################################################
print('============== Numerical parameters ===============')
print('Mesh resolution:', values_v.shape)
print('Time step:', ntime)
print('Initial time:', ctime)
print('Diagonal componet:', diag)
#######################################################
# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #
source_h = torch.zeros(input_shape, device=device)
max_flow = 0.0625 #2.0 #m/s
def get_source(real_time):
    global dt
    t = real_time/60   # minutes
    if t <= 5:
        source_h[0,0,192:208,196:204] = 0
    elif t > 5 and t < 60:
        source_h[0,0,192:208,196:204] = max_flow*(1/55*t - 1/11)/dx
        
    elif t < 4*60:
        source_h[0,0,192:208,196:204] = max_flow/dx
        
    elif t <= 5*60:
        source_h[0,0,192:208,196:204] = max_flow*(-1/60*t+5)/dx

point1 = torch.zeros((1,1000), device=device)
point3 = torch.zeros((1,1000), device=device)
point5 = torch.zeros((1,1000), device=device) 
point6 = torch.zeros((1,1000), device=device) 
time_x = torch.zeros((1,1000), device=device)

point1_u = torch.zeros((1,1000), device=device) 
point1_v = torch.zeros((1,1000), device=device)
point3_u = torch.zeros((1,1000), device=device)
point3_v = torch.zeros((1,1000), device=device)
point5_u = torch.zeros((1,1000), device=device)
point5_v = torch.zeros((1,1000), device=device)
point6_u = torch.zeros((1,1000), device=device)
point6_v = torch.zeros((1,1000), device=device)

point1_sigma = torch.zeros((1,1000), device=device)
point3_sigma = torch.zeros((1,1000), device=device)
point5_sigma = torch.zeros((1,1000), device=device)
point6_sigma = torch.zeros((1,1000), device=device)

# point1_u = []; point1_v = []
# point3_u = []; point3_v = []
# point5_u = []; point5_v = []
# point6_u = []; point6_v = []
    
def save_points(sigma_q, etha, u, v, hours, tt):
    x1 = int(1050//dx)   ; y1 = int(1000//dx)
    x3 = int(1200//dx)   ; y3 = int(1000//dx)
    x5 = int(1400//dx)   ; y5 = int(1000//dx)
    x6 = int(1300//dx)   ; y6 = int(1300//dx)


    time_x[0,tt] = hours
    
    point1[0,tt] = etha[y1,x1]
    point3[0,tt] = etha[y3,x3]
    point5[0,tt] = etha[y5,x5]
    point6[0,tt] = etha[y6,x6]
    
    point1_u[0,tt] = u[y1,x1] ; point1_v[0,tt] = v[y1,x1]
    point3_u[0,tt] = u[y3,x3] ; point3_v[0,tt] = v[y3,x3]
    point5_u[0,tt] = u[y5,x5] ; point5_v[0,tt] = v[y5,x5]
    point6_u[0,tt] = u[y6,x6] ; point6_v[0,tt] = v[y6,x6]
    point1_sigma[0,tt] = sigma_q[y1,x1]
    point3_sigma[0,tt] = sigma_q[y3,x3]
    point5_sigma[0,tt] = sigma_q[y5,x5]
    point6_sigma[0,tt] = sigma_q[y6,x6]

    
# Transfer array into tensor
# values_h = values_H
values_H = - values_H
# # # ################################### # # #
# # # #########   AI4SWE MAIN ########### # # #
# # # ################################### # # #
class AI4SWE(nn.Module):
    """docstring for two_step"""
    def __init__(self):
        super(AI4SWE, self).__init__()
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
# dbug         
        self.avg = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.cmm.weight.data = wm

        self.avg.weight.data = wavg

        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer

        self.avg.bias.data = bias_initializer

    def boundary_condition_u(self, values_u, values_uu):
        ny = values_u.shape[2]
        nx = values_u.shape[3]
        nny = values_uu.shape[2]
        nnx = values_uu.shape[3]

        values_uu[0,0,1:nny-1,1:nnx-1] = values_u[0,0,:,:]
        values_uu[0,0,:,0] =  values_uu[0,0,:,1]*0 
        values_uu[0,0,:,nx+1] = values_uu[0,0,:,nx]*0
        values_uu[0,0,0,:] = values_uu[0,0,1,:] 
        values_uu[0,0,ny+1,:] = values_uu[0,0,ny,:]
        return values_uu   

    def boundary_condition_v(self, values_v, values_vv):
        ny = values_v.shape[2]
        nx = values_v.shape[3]
        nny = values_vv.shape[2]
        nnx = values_vv.shape[3]

        values_vv[0,0,1:nny-1,1:nnx-1] = values_v[0,0,:,:]
        values_vv[0,0,:,0] =  values_vv[0,0,:,1]
        values_vv[0,0,:,nx+1] = values_vv[0,0,:,nx]
        values_vv[0,0,0,:] = values_vv[0,0,1,:]*0
        values_vv[0,0,ny+1,:] = values_vv[0,0,ny,:]*0
        return values_vv       

    def boundary_condition_eta(self, values_h, values_hp):
        ny = values_h.shape[2]
        nx = values_h.shape[3]
        nny = values_hp.shape[2]
        nnx = values_hp.shape[3]

        values_hp[0,0,1:nny-1,1:nnx-1] = values_h[0,0,:,:]
        values_hp[0,0,:,nx+1] = values_hp[0,0,:,nx]      
        values_hp[0,0,:,0] = values_hp[0,0,:,1] 
        values_hp[0,0,ny+1,:] = values_hp[0,0,ny,:]  
        values_hp[0,0,0,:] = values_hp[0,0,1,:] 
        return values_hp   

    def PG_vector(self, values_uu, values_vv, values_u, values_v, k3):
        k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_uu)) / \
            (1e-03  + (torch.abs(self.xadv(values_uu)) * (dx**-2) + torch.abs(self.yadv(values_uu)) * (dx**-2)) / 2)

        k_v = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_vv)) / \
            (1e-03  + (torch.abs(self.xadv(values_vv)) * (dx**-2) + torch.abs(self.yadv(values_vv)) * (dx**-2)) / 2)

        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(torch.minimum(k_v, k3) , (1, 1, 1, 1), mode='constant', value=0)

        k_x = 0.5 * (k_u * self.diff(values_uu) + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * self.diff(values_vv) + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        return k_x, k_y

    def PG_scalar(self, values_hh, values_h, values_u, values_v, k3):
        k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_hh)) / \
            (1e-03 + (torch.abs(self.xadv(values_hh)) * (dx**-2) + torch.abs(self.yadv(values_hh)) * (dx**-2)) / 2)  
        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        return 0.5 * (k_u * self.diff(values_hh) + self.diff(values_hh * k_uu) - values_h * self.diff(k_uu))        

    def forward(self, values_u, values_uu, values_v, values_vv, values_H, values_h, values_hp, b_u, b_uu, b_v, b_vv, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp):
        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)

        [k_x,k_y] = self.PG_vector(values_uu, values_vv, values_u, values_v, k3)
        b_u = (k_x * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt) * 0.5 + values_u
        b_v = (k_y * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt) * 0.5 + values_v
        b_u = b_u - self.xadv(self.boundary_condition_eta(values_h,values_hp)) * dt
        b_v = b_v - self.yadv(self.boundary_condition_eta(values_h,values_hp)) * dt

        b_uu = self.boundary_condition_u(b_u,b_uu)       
        b_vv = self.boundary_condition_v(b_v,b_vv)

        # sigma_q = self.cmm((self.boundary_condition_u(b_u,b_uu)**2 + self.boundary_condition_v(b_v,b_vv)**2)**0.5) / \
        # (torch.maximum(k1*10,(5*self.cmm(self.boundary_condition_eta(values_h,values_hp))))**(4/3))

        # sigma_q = (b_u**2 + b_v**2)**0.5 * 0.05**2 / (torch.maximum(k1,torch.maximum(values_h,
        # dx*self.cmm(self.boundary_condition_eta(values_h,values_hp))))**(4/3)) #getting working but still not good

        # sigma_q = (b_u**2 + b_v**2)**0.5 * 0.05**2 / (torch.maximum(k1*10,(values_H+values_h)*1.0)**(4/3)) #not working 

        # sigma_q = (b_u**2 + b_v**2)**0.5 * 0.05**2 / (torch.maximum(k1*10000,(values_H+values_h))**(4/3)) working but not reasonable 

        sigma_q = (b_u**2 + b_v**2)**0.5 * 0.05**2 / (torch.maximum(k1,torch.maximum( values_h,
                dx*self.cmm(self.boundary_condition_eta(values_h,values_hp))*1.0 + values_h*0.0 ))**(4/3)) #g
        # b_u = torch.div(b_u, 1 + torch.div(torch.mul(sigma_q, dt),rho))
        # b_v = torch.div(b_v, 1 + torch.div(torch.mul(sigma_q, dt),rho))
        b_u = b_u / (1 + sigma_q * dt / rho)
        b_v = b_v / (1 + sigma_q * dt / rho)

        [k_x,k_y] = self.PG_vector(b_uu, b_vv, b_u, b_v, k3)
        values_u = values_u + k_x * dt - b_u * self.xadv(b_uu) * dt - b_v * self.yadv(b_uu) * dt   
        values_v = values_v + k_y * dt - b_u * self.xadv(b_vv) * dt - b_v * self.yadv(b_vv) * dt 
        values_u = values_u - self.xadv(self.boundary_condition_eta(values_h,values_hp)) * dt
        values_v = values_v - self.yadv(self.boundary_condition_eta(values_h,values_hp)) * dt      

        # sigma_q = self.cmm((self.boundary_condition_u(values_u,values_uu)**2 + self.boundary_condition_v(values_v,values_vv)**2)**0.5) / \
        # (torch.maximum(k1*10,(5*self.cmm(self.boundary_condition_eta(values_h,values_hp))))**(4/3)) crazying !

        # sigma_q = (values_u**2 + values_v**2)**0.5 * 0.05**2 / (torch.maximum(k1,torch.maximum(values_h,
        #     dx*self.cmm(self.boundary_condition_eta(values_h,values_hp))))**(4/3))

        # sigma_q = (values_u**2 + values_v**2)**0.5 * 0.05**2 / (torch.maximum(k1*10,(values_H+values_h)*1.0)**(4/3))

        # sigma_q = (values_u**2 + values_v**2)**0.5 * 0.05**2 / (torch.maximum(k1*10000,(values_H+values_h))**(4/3))

        sigma_q = (values_u**2 + values_v**2)**0.5 * 0.05**2 / (torch.maximum(k1,torch.maximum(values_h,
            dx*self.cmm(self.boundary_condition_eta(values_h,values_hp))*1.0 + values_h*0.0           ))**(4/3))

        values_u = values_u / (1 + sigma_q * dt / rho)
        values_v = values_v / (1 + sigma_q * dt / rho)
        # values_u = torch.div(values_u, 1 + torch.div(torch.mul(sigma_q, dt),rho))
        # values_v = torch.div(values_v, 1 + torch.div(torch.mul(sigma_q, dt),rho))
        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)
        eta1 = torch.maximum(k2,(values_H+values_h))
        eta2 = torch.maximum(k1,(values_H+values_h))
        # dbug = 
        b = beta * rho * (-self.xadv(self.boundary_condition_eta(eta1,eta1_p)) * values_u - \
                           self.yadv(self.boundary_condition_eta(eta1,eta1_p)) * values_v - \
                           eta1 * self.xadv(values_uu) - eta1 * self.yadv(values_vv) + \
                           self.PG_scalar(self.boundary_condition_eta(eta1,eta1_p), eta1, values_u, values_v, k3) - \
                           self.cmm(self.boundary_condition_eta(dif_values_h,dif_values_hh)) / dt + self.cmm(self.boundary_condition_eta(source_h,dif_values_hh))) / (dt * eta2)   

                           # self.cmm(self.boundary_condition_eta(dif_values_h,dif_values_hh)) / dt + source_h) / (dt * eta2)   
        values_h_old = values_h.clone()
        for i in range(2):
            values_hh = values_hh - (-self.diff(self.boundary_condition_eta(values_hh,values_hhp)) + beta * rho / (dt**2 * eta2) * values_hh) / \
                        (diag + beta * rho / (dt**2 * eta2)) + b / (diag + beta * rho / (dt**2 * eta2))
        values_h = values_h + values_hh
        dif_values_h = values_h - values_h_old 
        values_u = values_u - self.xadv(self.boundary_condition_eta(values_hh,values_hhp)) * dt / rho
        values_v = values_v - self.yadv(self.boundary_condition_eta(values_hh,values_hhp)) * dt / rho 

        return values_u, values_v, values_h, values_hh, b, dif_values_h, sigma_q

model = AI4SWE().to(device)

start = time.time()
with torch.no_grad():
    for itime in range(1,ntime+1):
        get_source(real_time)
        for t in range(2):
            [values_u, values_v, values_h, values_hh, b, dif_values_h, sigma_q] = model(values_u, values_uu, values_v, values_vv, values_H, values_h, 
                    values_hp, b_u, b_uu, b_v, b_vv, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp)
# output              
        real_time = real_time + dt

        print('Time step:', itime) 
        print('height correction:', np.max(values_hh.cpu().detach().numpy()))
        print('========================================================')
        if np.max(np.abs(values_hh.cpu().detach().numpy())) > 10.0:
            print('Not converged !!!!!!')
            # np.save("temp/H"+str(itime), arr=values_H.cpu().detach().numpy()[0,0,:,:])
            # np.save("temp/h"+str(itime), arr=values_h.cpu().detach().numpy()[0,0,:,:])
            # np.save("temp/u"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
            # np.save("temp/v"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])
            # np.save("temp/b"+str(itime), arr=b.cpu().detach().numpy()[0,0,:,:])
            # np.save("temp/dh"+str(itime), arr=dif_values_h.cpu().detach().numpy()[0,0,:,:])
            break
        if istep % 120 ==0:
            tt += 1
            print('Time step:', itime, istep, 'time =', real_time/60)
            save_points((sigma_q)[0,0,:,:], (values_h)[0,0,:,:], values_u[0,0,:,:], values_v[0,0,:,:], real_time, tt)
        
        istep +=1

        if save_fig == True and itime == ntime:
            # print(point1)
            # torch.save(point1_cpu, 'point1.npy')
            # print(np.array(point1).cpu().shape)
            filepath = 'Data_flooding_validation/'

            np.save(filepath+"point1", arr=point1.cpu().detach().numpy())
            np.save(filepath+"point3", arr=point3.cpu().detach().numpy())
            np.save(filepath+"point5", arr=point5.cpu().detach().numpy())
            np.save(filepath+"point6", arr=point6.cpu().detach().numpy())

            np.save(filepath+"point1_u", arr=point1_u.cpu().detach().numpy())
            np.save(filepath+"point3_u", arr=point3_u.cpu().detach().numpy())
            np.save(filepath+"point5_u", arr=point5_u.cpu().detach().numpy())
            np.save(filepath+"point6_u", arr=point6_u.cpu().detach().numpy())

            np.save(filepath+"point1_v", arr=point1_v.cpu().detach().numpy())
            np.save(filepath+"point3_v", arr=point3_v.cpu().detach().numpy())
            np.save(filepath+"point5_v", arr=point5_v.cpu().detach().numpy())
            np.save(filepath+"point6_v", arr=point6_v.cpu().detach().numpy())

            np.save(filepath+"point1_s", arr=point1_sigma.cpu().detach().numpy())
            np.save(filepath+"point3_s", arr=point3_sigma.cpu().detach().numpy())
            np.save(filepath+"point5_s", arr=point5_sigma.cpu().detach().numpy())
            np.save(filepath+"point6_s", arr=point6_sigma.cpu().detach().numpy())

            np.save(filepath+"time", arr=time_x.cpu().detach().numpy())

            print(tt)
            # np.save("point3", arr=np.array(point3))
            # np.save("point5", arr=np.array(point5))
            # np.save("point6", arr=np.array(point6))
            # np.save("time", arr=np.array(time_x))
        if itime == 7200 or itime == 21600 or itime == 36000:
            filepath = 'Data_flooding_validation/'
            np.save(filepath+"one"+str(itime), arr=values_h.cpu().detach().numpy()[0,0,:,:])


    end = time.time()
    print('time',(end-start))
