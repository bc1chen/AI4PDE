#!/usr/bin/env python

#  Copyright (C) 2023
#  
#  Boyang Chen, Claire Heaney, Christopher Pain
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
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
nx = 512
ny = 512
nz = 64
dx = 1.0 ; dy = 1.0 ; dz = 1.0
Re = 0.15
dt = 0.5
ub = -1.0
# ratio = int(max(nx, ny, nz) / min(nx, ny, nz))
ratio_x = int(nx/nz)
ratio_y = int(ny/nz)
nlevel = int(math.log(min(nx, ny, nz), 2)) + 1 
print('How many levels in multigrid:', nlevel)
print('Aspect ratio:', ratio_x)
print('Grid spacing:', dx)
# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0])
# Laplacian filters
pd1 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
pd2 = torch.tensor([[3/26, 6/26, 3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26, 3/26]])
pd3 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
w1 = torch.zeros([1, 1, 3, 3, 3])
wA = torch.zeros([1, 1, 3, 3, 3])
w1[0, 0, 0,:,:] = pd1/dx**2
w1[0, 0, 1,:,:] = pd2/dx**2
w1[0, 0, 2,:,:] = pd3/dx**2
wA[0, 0, 0,:,:] = -pd1/dx**2
wA[0, 0, 1,:,:] = -pd2/dx**2
wA[0, 0, 2,:,:] = -pd3/dx**2
# Gradient filters
p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056],
       [-0.22, 0.0, 0.22],
       [-0.056, 0.0, 0.056]])
p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_y1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_y2 = torch.tensor([[0.056, 0.22, 0.056],
       [0.0, 0.0, 0.0],
       [-0.056, -0.22, -0.056]])
p_div_y3 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.056, 0.22, 0.056],
       [0.014, 0.056, 0.014]])
p_div_z2 = torch.tensor([[0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0]])
p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014],
       [-0.056, -0.22, -0.056],
       [-0.014, -0.056, -0.014]])
w2 = torch.zeros([1,1,3,3,3])
w3 = torch.zeros([1,1,3,3,3])
w4 = torch.zeros([1,1,3,3,3])
w2[0,0,0,:,:] = -p_div_x1/dx*0.5
w2[0,0,1,:,:] = -p_div_x2/dx*0.5
w2[0,0,2,:,:] = -p_div_x3/dx*0.5
w3[0,0,0,:,:] = -p_div_y1/dx*0.5
w3[0,0,1,:,:] = -p_div_y2/dx*0.5
w3[0,0,2,:,:] = -p_div_y3/dx*0.5
w4[0,0,0,:,:] = -p_div_z1/dx*0.5
w4[0,0,1,:,:] = -p_div_z2/dx*0.5
w4[0,0,2,:,:] = -p_div_z3/dx*0.5
# Restriction filters
w_res = torch.zeros([1,1,2,2,2])
w_res[0,0,:,:,:] = 0.125
################# Numerical parameters ################
ntime = 10000                     # Time steps
n_out = 2000                       # Results output
iteration = 5                    # Multigrid iteration
nrestart = 0                      # Last time step for restart
ctime_old = 0                     # Last ctime for restart
LIBM = True                      # Immersed boundary method 
ctime = 0                         # Initialise ctime   
save_fig = True                   # Save results
Restart = False                   # Restart
eplsion_k = 1e-04                 # Stablisatin factor in Petrov-Galerkin for velocity
diag = np.array(wA)[0,0,1,1,1]    # Diagonal component
#######################################################
# # # ################################### # # #
# # # ######    Create tensor      ###### # # #
# # # ################################### # # #
input_shape = (1,1,nz,ny,nx)
values_u = torch.zeros(input_shape, device=device)
values_v = torch.zeros(input_shape, device=device)
values_w = torch.zeros(input_shape, device=device)
values_p = torch.zeros(input_shape, device=device)
k1 = torch.ones(input_shape, device=device)*2.0
input_shape_pad = (1,1,nz+2,ny+2,nx+2)
values_uu = torch.zeros(input_shape_pad, device=device)
values_vv = torch.zeros(input_shape_pad, device=device)
values_ww = torch.zeros(input_shape_pad, device=device)
values_pp = torch.zeros(input_shape_pad, device=device)
b_uu = torch.zeros(input_shape_pad, device=device)
b_vv = torch.zeros(input_shape_pad, device=device)
b_ww = torch.zeros(input_shape_pad, device=device)
k_uu = torch.zeros(input_shape_pad, device=device)
k_vv = torch.zeros(input_shape_pad, device=device)
k_ww = torch.zeros(input_shape_pad, device=device)
#######################################################
print('============== Numerical parameters ===============')
print('Mesh resolution:', values_v.shape)
print('Time step:', ntime)
print('Initial time:', ctime)
print('Diagonal componet:', diag)
#######################################################
################# Only for restart ####################
if Restart == True:
    nrestart = 8000
    ctime_old = nrestart*dt
    print('Restart solver!')
# #######################################################    
# ################# Only for IBM ########################
if LIBM == True:
    mesh = np.load("INHALE_1280.npy")
    sigma = torch.zeros(input_shape, dtype=torch.float32, device=device) 
    print(mesh.shape, sigma.shape)
    for i in range(nz):
        sigma[0,0,i,:,:] = torch.tensor(mesh[0,180:692,240:752,i,0])
    sigma = sigma.transpose_(4, 3)
    sigma = torch.flip(sigma, [3])
    sigma = torch.where(sigma == 0, torch.tensor(1e08, dtype=torch.float32, device=device), torch.tensor(0, dtype=torch.float32, device=device))
    plt.imshow(sigma[0,0,4,:,:].cpu())
    plt.colorbar()
    plt.savefig('South_Kensington.jpg')
    plt.close()
#######################################################
# # # ################################### # # #
# # # #########  AI4Urban MAIN ########## # # #
# # # ################################### # # #
class AI4Urban(nn.Module):
    """docstring for AI4Urban"""
    def __init__(self):
        super(AI4Urban, self).__init__()
        # self.arg = arg
        self.xadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.zadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.A = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)

        self.A.weight.data = wA
        self.res.weight.data = w_res
        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.zadv.weight.data = w4

        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.zadv.bias.data = bias_initializer
###############################################################
    def boundary_condition_u(self, values_u, values_uu):
        nz = values_u.shape[2]
        ny = values_u.shape[3]
        nx = values_u.shape[4]
        nnz = values_uu.shape[2]
        nny = values_uu.shape[3]
        nnx = values_uu.shape[4]

        values_uu[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_u[0,0,:,:,:]        

        values_uu[0,0,:,:,0].fill_(ub)
        values_uu[0,0,:,:,nx+1].fill_(ub)
        values_uu[0,0,:,0,:] = values_uu[0,0,:,1,:]
        values_uu[0,0,:,ny+1,:] = values_uu[0,0,:,ny,:]
        values_uu[0,0,0,:,:] = values_uu[0,0,1,:,:]*0 
        values_uu[0,0,nz+1,:,:] = values_uu[0,0,nz,:,:]
        return values_uu

    def boundary_condition_v(self, values_v, values_vv):
        nz = values_v.shape[2]
        ny = values_v.shape[3]
        nx = values_v.shape[4]
        nnz = values_vv.shape[2]
        nny = values_vv.shape[3]
        nnx = values_vv.shape[4]

        values_vv[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_v[0,0,:,:,:]

        values_vv[0,0,:,:,0] = values_vv[0,0,:,:,1]*0 
        values_vv[0,0,:,:,nx+1] = values_vv[0,0,:,:,nx]*0
        values_vv[0,0,:,0,:].fill_(0.0)
        values_vv[0,0,:,ny+1,:].fill_(0.0)
        values_vv[0,0,0,:,:] = values_vv[0,0,1,:,:]*0
        values_vv[0,0,nz+1,:,:] = values_vv[0,0,nz,:,:]
        return values_vv

    def boundary_condition_w(self, values_w, values_ww):
        nz = values_w.shape[2]
        ny = values_w.shape[3]
        nx = values_w.shape[4]
        nnz = values_ww.shape[2]
        nny = values_ww.shape[3]
        nnx = values_ww.shape[4]

        values_ww[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_w[0,0,:,:,:]
        
        values_ww[0,0,:,:,0] =  values_ww[0,0,:,:,1]*0 
        values_ww[0,0,:,:,nx+1] = values_ww[0,0,:,:,nx]*0
        values_ww[0,0,:,0,:] = values_ww[0,0,:,1,:]*0
        values_ww[0,0,:,ny+1,:] = values_ww[0,0,:,ny,:]*0
        values_ww[0,0,0,:,:].fill_(0.0)
        values_ww[0,0,nz+1,:,:].fill_(0.0)
        return values_ww

    def solid_body(self, values_u, values_v, values_w, sigma, dt):
        values_u = values_u / (1+dt*sigma) 
        values_v = values_v / (1+dt*sigma) 
        values_w = values_w / (1+dt*sigma) 
        return values_u, values_v, values_w

    def boundary_condition_p(self, values_p, values_pp):  # need check this boundary condition for the real case 
        nz = values_p.shape[2]
        ny = values_p.shape[3]
        nx = values_p.shape[4]
        nnz = values_pp.shape[2]
        nny = values_pp.shape[3]
        nnx = values_pp.shape[4]
        
        values_pp[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_p[0,0,:,:,:]

        values_pp[0,0,:,:,0] =  values_pp[0,0,:,:,1] 
        values_pp[0,0,:,:,nx+1] = values_pp[0,0,:,:,nx]*0 # outflow boundary condition 
        values_pp[0,0,:,0,:] = values_pp[0,0,:,1,:]
        values_pp[0,0,:,ny+1,:] = values_pp[0,0,:,ny,:]
        values_pp[0,0,0,:,:] = values_pp[0,0,1,:,:]
        values_pp[0,0,nz+1,:,:] = values_pp[0,0,nz,:,:]
        return values_pp

    def boundary_condition_k(self, k_u, k_uu):
        nz = k_u.shape[2]
        ny = k_u.shape[3]
        nx = k_u.shape[4]
        nnz = k_uu.shape[2]
        nny = k_uu.shape[3]
        nnx = k_uu.shape[4]

        k_uu[0,0,1:nnz-1,1:nny-1,1:nnx-1] = k_u[0,0,:,:,:]
        k_uu[0,0,:,:,0] =  k_uu[0,0,:,:,1]*0 
        k_uu[0,0,:,:,nx+1] = k_uu[0,0,:,:,nx]*0 
        k_uu[0,0,:,0,:] = k_uu[0,0,:,1,:]*0
        k_uu[0,0,:,ny+1,:] = k_uu[0,0,:,ny,:]*0
        k_uu[0,0,0,:,:] = k_uu[0,0,1,:,:]*0
        k_uu[0,0,nz+1,:,:] = k_uu[0,0,nz,:,:]*0
        return k_uu

    def boundary_condition_cw(self, w):
        nz = w.shape[2]
        ny = w.shape[3]
        nx = w.shape[4]
        ww = F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    
        ww[0,0,:,:,0] =  ww[0,0,:,:,1]*0 
        ww[0,0,:,:,nx+1] = ww[0,0,:,:,nx]*0 
        ww[0,0,:,0,:] = ww[0,0,:,1,:]*0
        ww[0,0,:,ny+1,:] = ww[0,0,:,ny,:]*0
        ww[0,0,0,:,:] = ww[0,0,1,:,:]*0
        ww[0,0,nz+1,:,:] = ww[0,0,nz,:,:]*0
        return ww

    def F_cycle_MG(self, values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y):
        b = -(self.xadv(values_uu) + self.yadv(values_vv) + self.zadv(values_ww)) / dt
        for MG in range(iteration):
              w = torch.zeros((1,1,2,2*ratio_y,2*ratio_x), device=device)
              ww = torch.zeros((1,1,2+2,2*ratio_y+2,2*ratio_x+2), device=device)
              r = self.A(self.boundary_condition_p(values_p, values_pp)) - b 
              r_s = []  
              r_s.append(r)
              for i in range(1,nlevel-1):
                     r = self.res(r)
                     r_s.append(r)
              for i in reversed(range(1,nlevel-1)):
                     ww = self.boundary_condition_cw(w)
                     w = w - self.A(ww) / diag + r_s[i] / diag
                     w = self.prol(w)         
              values_p = values_p - w
              values_p = values_p - self.A(self.boundary_condition_p(values_p, values_pp)) / diag + b / diag
        return values_p, w, r

    def PG_vector(self, values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w):
        k_u = 0.1 * dx * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_u) / \
            (1e-03 + (torch.abs(ADx_u) * dx**-3 + torch.abs(ADy_u) * dx**-3 + torch.abs(ADz_u) * dx**-3) / 3)

        k_v = 0.1 * dy * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_v) / \
            (1e-03 + (torch.abs(ADx_v) * dx**-3 + torch.abs(ADy_v) * dx**-3 + torch.abs(ADz_v) * dx**-3) / 3)

        k_w = 0.1 * dz * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_w) / \
            (1e-03 + (torch.abs(ADx_w) * dx**-3 + torch.abs(ADy_w) * dx**-3 + torch.abs(ADz_w) * dx**-3) / 3)

        k_u = torch.minimum(k_u, k1) / (1+dt*sigma) 
        k_v = torch.minimum(k_v, k1) / (1+dt*sigma) 
        k_w = torch.minimum(k_w, k1) / (1+dt*sigma) 

        k_uu = self.boundary_condition_k(k_u,k_uu)     # **************************** halo update -> k_uu ****************************
        k_vv = self.boundary_condition_k(k_v,k_vv)     # **************************** halo update -> k_vv ****************************
        k_ww = self.boundary_condition_k(k_w,k_ww)     # **************************** halo update -> k_ww ****************************

        k_x = 0.5 * (k_u * AD2_u + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * AD2_v + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        k_z = 0.5 * (k_w * AD2_w + self.diff(values_ww * k_ww) - values_w * self.diff(k_ww))
        return k_x, k_y, k_z

    def forward(self, values_u, values_uu, values_v, values_vv, values_w, values_ww, values_p, values_pp, b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww):
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
    # Padding velocity vectors 
        values_uu = self.boundary_condition_u(values_u,values_uu) # ****************************** halo update -> values_uu ************************  
        values_vv = self.boundary_condition_v(values_v,values_vv) # ****************************** halo update -> values_vv ************************ 
        values_ww = self.boundary_condition_w(values_w,values_ww) # ****************************** halo update -> values_ww ************************ 
        values_pp = self.boundary_condition_p(values_p,values_pp) # ****************************** halo update -> values_pp ************************ 

        Grapx_p = self.xadv(values_pp) * dt ; Grapy_p = self.yadv(values_pp) * dt ; Grapz_p = self.zadv(values_pp) * dt  
        ADx_u = self.xadv(values_uu) ; ADy_u = self.yadv(values_uu) ; ADz_u = self.zadv(values_uu)
        ADx_v = self.xadv(values_vv) ; ADy_v = self.yadv(values_vv) ; ADz_v = self.zadv(values_vv)
        ADx_w = self.xadv(values_ww) ; ADy_w = self.yadv(values_ww) ; ADz_w = self.zadv(values_ww)
        AD2_u = self.diff(values_uu) ; AD2_v = self.diff(values_vv) ; AD2_w = self.diff(values_ww)
    # First step for solving uvw
        [k_x,k_y,k_z] = self.PG_vector(values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, 
                                       ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w)

        b_u = values_u + 0.5 * (Re * k_x * dt - values_u * ADx_u * dt - values_v * ADy_u * dt - values_w * ADz_u * dt) - Grapx_p
        b_v = values_v + 0.5 * (Re * k_y * dt - values_u * ADx_v * dt - values_v * ADy_v * dt - values_w * ADz_v * dt) - Grapy_p
        b_w = values_w + 0.5 * (Re * k_z * dt - values_u * ADx_w * dt - values_v * ADy_w * dt - values_w * ADz_w * dt) - Grapz_p
    # Solid body
        if LIBM == True: [b_u, b_v, b_w] = self.solid_body(b_u, b_v, b_w, sigma, dt)
    # Padding velocity vectors 
        b_uu = self.boundary_condition_u(b_u,b_uu) # ****************************** halo update -> b_uu ************************ 
        b_vv = self.boundary_condition_v(b_v,b_vv) # ****************************** halo update -> b_vv ************************ 
        b_ww = self.boundary_condition_w(b_w,b_ww) # ****************************** halo update -> b_ww ************************ 

        ADx_u = self.xadv(b_uu) ; ADy_u = self.yadv(b_uu) ; ADz_u = self.zadv(b_uu)
        ADx_v = self.xadv(b_vv) ; ADy_v = self.yadv(b_vv) ; ADz_v = self.zadv(b_vv)
        ADx_w = self.xadv(b_ww) ; ADy_w = self.yadv(b_ww) ; ADz_w = self.zadv(b_ww)
        AD2_u = self.diff(b_uu) ; AD2_v = self.diff(b_vv) ; AD2_w = self.diff(b_ww)

        [k_x,k_y,k_z] = self.PG_vector(b_uu, b_vv, b_ww, b_u, b_v, b_w, k1, k_uu, k_vv, k_ww, 
                                       ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w)   
    # Second step for solving uvw   
        values_u = values_u + Re * k_x * dt - b_u * ADx_u * dt - b_v * ADy_u * dt - b_w * ADz_u * dt - Grapx_p  
        values_v = values_v + Re * k_y * dt - b_u * ADx_v * dt - b_v * ADy_v * dt - b_w * ADz_v * dt - Grapy_p  
        values_w = values_w + Re * k_z * dt - b_u * ADx_w * dt - b_v * ADy_w * dt - b_w * ADz_w * dt - Grapz_p
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
    # pressure
        values_uu = self.boundary_condition_u(values_u,values_uu) # ****************************** halo update -> values_uu ************************ 
        values_vv = self.boundary_condition_v(values_v,values_vv) # ****************************** halo update -> values_vv ************************ 
        values_ww = self.boundary_condition_w(values_w,values_ww) # ****************************** halo update -> values_ww ************************
        [values_p, w ,r] = self.F_cycle_MG(values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y)
    # Pressure gradient correction    
        values_pp = self.boundary_condition_p(values_p, values_pp) # ****************************** halo update -> values_pp ************************       
        values_u = values_u - self.xadv(values_pp) * dt ; values_v = values_v - self.yadv(values_pp) * dt ; values_w = values_w - self.zadv(values_pp) * dt      
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
        return values_u, values_v, values_w, values_p, w, r

model = AI4Urban().to(device)

start = time.time()
with torch.no_grad():
    for itime in range(1,ntime+1):
        [values_u, values_v, values_w, values_p, w, r] = model(values_u, values_uu, values_v, values_vv, values_w, values_ww, values_p, values_pp, b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww)
# output   
        print('Time step:', itime) 
        print('Pressure error:', np.max(np.abs(w.cpu().detach().numpy())), 'cty equation residual:', np.max(np.abs(r.cpu().detach().numpy())))
        print('========================================================')
        if np.max(np.abs(w.cpu().detach().numpy())) > 80000.0:
            print('Not converged !!!!!!')
            break
        if save_fig == True and itime % n_out == 0:
            np.save("Urban_data_L_small/w"+str(itime), arr=values_w.cpu().detach().numpy()[0,0,:,:])
            np.save("Urban_data_L_small/v"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
            np.save("Urban_data_L_small/u"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])
            # np.save("temp1/p"+str(itime), arr=values_p.cpu().detach().numpy()[0,0,:,:])
    end = time.time()
    print('time',(end-start))
