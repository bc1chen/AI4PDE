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

# Specify the file path that you want to save your results
save_path = 'Stokes_L'
# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
nx = 512 ; ny = 512 ; nz = 480
dx = 0.0001 ; dy = 0.0001 ; dz = 0.0001 ; dt = 0.0001
det = dx*dx*dz
ratio = int(max(nx, ny, nz) / min(nx, ny, nz))
nlevel = int(math.log(min(nx, ny, nz), 2)) + 1 
print('How many levels in multigrid:', nlevel)
print('Aspect ratio:', ratio)
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
w1 = torch.zeros([1, 1, 3, 3, 3]) ; wA = torch.zeros([1, 1, 3, 3, 3])
w1[0, 0, 0,:,:] = pd1/dx**2  ; w1[0, 0, 1,:,:] = pd2/dx**2  ; w1[0, 0, 2,:,:] = pd3/dx**2
wA[0, 0, 0,:,:] = -pd1/dx**2 ; wA[0, 0, 1,:,:] = -pd2/dx**2 ; wA[0, 0, 2,:,:] = -pd3/dx**2
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
w2 = torch.zeros([1,1,3,3,3]) ; w3 = torch.zeros([1,1,3,3,3]) ; w4 = torch.zeros([1,1,3,3,3])
w2[0,0,0,:,:] = -p_div_x1/dx ; w2[0,0,1,:,:] = -p_div_x2/dx ; w2[0,0,2,:,:] = -p_div_x3/dx
w3[0,0,0,:,:] = -p_div_y1/dx ; w3[0,0,1,:,:] = -p_div_y2/dx ; w3[0,0,2,:,:] = -p_div_y3/dx
w4[0,0,0,:,:] = -p_div_z1/dx ; w4[0,0,1,:,:] = -p_div_z2/dx ; w4[0,0,2,:,:] = -p_div_z3/dx
# Curvature Laplacian filters
curvature_x1 = torch.tensor([[-0.1875, 0.375,  -0.1875],
       [-0.75, 1.5,  -0.75],
       [-0.1875, 0.375,  -0.1875]])
curvature_x2= torch.tensor([[-0.75, 1.5,  -0.75],
       [-3.0, 6.0,  -3.0],
       [-0.75, 1.5,  -0.75]])
curvature_x3 = torch.tensor([[-0.1875, 0.375,  -0.1875],
       [-0.75, 1.5,  -0.75],
       [-0.1875, 0.375,  -0.1875]])
curvature_y1 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [0.375, 1.5,  0.375],
       [-0.1875, -0.75,  -0.1875]])
curvature_y2= torch.tensor([[-0.75, -3.0,  -0.75],
       [1.5, 6.0,  1.5],
       [-0.75, -3.0,  -0.75]])
curvature_y3 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [0.375, 1.5,  0.375],
       [-0.1875, -0.75,  -0.1875]])
curvature_z1 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [-0.75, -3.0,  -0.75],
       [-0.1875, -0.75,  -0.1875]])
curvature_z2= torch.tensor([[0.375, 1.5,  0.375],
       [1.5, 6.0,  1.5],
       [0.375, 1.5,  0.375]])
curvature_z3 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [-0.75, -3.0,  -0.75],
       [-0.1875, -0.75,  -0.1875]])
AD2_x = torch.zeros([1,1,3,3,3]) ; AD2_y = torch.zeros([1,1,3,3,3]) ; AD2_z = torch.zeros([1,1,3,3,3])
AD2_x[0,0,0,:,:] = -curvature_x1/dx**2 ; AD2_x[0,0,1,:,:] = -curvature_x2/dx**2 ; AD2_x[0,0,2,:,:] = -curvature_x3/dx**2
AD2_y[0,0,0,:,:] = -curvature_y1/dx**2 ; AD2_y[0,0,1,:,:] = -curvature_y2/dx**2 ; AD2_y[0,0,2,:,:] = -curvature_y3/dx**2
AD2_z[0,0,0,:,:] = -curvature_z1/dx**2 ; AD2_z[0,0,1,:,:] = -curvature_z2/dx**2 ; AD2_z[0,0,2,:,:] = -curvature_z3/dx**2
# Restriction filters
w_res = torch.zeros([1,1,2,2,2]) ; w_res[0,0,:,:,:] = 0.125
################# Numerical parameters ################
ntime = 20000                    # Time steps
n_out = 2000                     # Results output
iteration = 5                     # Multigrid iteration
nrestart = 0                      # Last time step for restart
ctime_old = 0                     # Last ctime for restart
LSCALAR = True                    # Scalar transport 
LMTI = True                       # Non density for multiphase flows
LIBM = True                       # Immersed boundary method 
ctime = 0                         # Initialise ctime   
save_fig = True                   # Save results
Restart = False                   # Restart
eplsion_k = 1e-04                 # Stablisatin factor in Petrov-Galerkin for velocity
real_time = 0
istep = 0 
################# Physical parameters #################
rho_l = 1000                      # Density of liquid phase 
rho_g = 1.0                       # Density of gas phase 
g_x = 0;g_y = 0;g_z = -10         # Gravity acceleration (m/s2) 
diag = np.array(wA)[0,0,1,1,1]    # Diagonal component
#######################################################
# # # ################################### # # #
# # # ######    Create tensor      ###### # # #
# # # ################################### # # #
input_shape = (1,1,nz,ny,nx)
input_shape_pad = (1,1,nz+2,ny+2,nx+2)
values_u = torch.zeros(input_shape, device=device)      ; values_v = torch.zeros(input_shape, device=device)      ; values_w = torch.zeros(input_shape, device=device)
values_uu = torch.zeros(input_shape_pad, device=device) ; values_vv = torch.zeros(input_shape_pad, device=device) ; values_ww = torch.zeros(input_shape_pad, device=device)
b_u = torch.zeros(input_shape, device=device)      ; b_v = torch.zeros(input_shape, device=device)      ; b_w = torch.zeros(input_shape, device=device)
b_uu = torch.zeros(input_shape_pad, device=device) ; b_vv = torch.zeros(input_shape_pad, device=device) ; b_ww = torch.zeros(input_shape_pad, device=device)
values_ph = torch.zeros(input_shape, device=device)      ; values_pd = torch.zeros(input_shape, device=device)
values_phh = torch.zeros(input_shape_pad, device=device) ; values_pdd = torch.zeros(input_shape_pad, device=device)
alpha = torch.zeros(input_shape, device=device) ; alphaa = torch.zeros(input_shape_pad, device=device)
k1 = torch.ones(input_shape, device=device)
k2 = torch.zeros(input_shape, device=device)
k3 = torch.ones(input_shape, device=device)*-1.0
k4 = torch.ones(input_shape, device=device)*dx**2*0.25/dt
k5 = torch.ones(input_shape, device=device)*dx**2*0.05/dt
k6 = torch.ones(input_shape, device=device)*dx**2*-0.0001/dt
rhoo = torch.zeros(input_shape_pad, device=device)
k7 = torch.ones(input_shape_pad, device=device)
k8 = torch.zeros(input_shape_pad, device=device)

nz = values_u.shape[2]
ny = values_u.shape[3]
nx = values_u.shape[4]
nnz = values_uu.shape[2]
nny = values_uu.shape[3]
nnx = values_uu.shape[4]
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
#######################################################    
################# Only for scalar #####################
if LSCALAR == True and Restart == False:
    alpha = torch.ones(input_shape, dtype=torch.float32, device=device) 
    # please use this line if not injecting a source with dt !!!!!!!!!!!!!!!!!!!!!!!
    # alpha[0,0,0:int(0.75//dz),:,0:int(15.5//dz)].fill_(1.0)
    # alpha[0,0,0:2,390:410,38:42].fill_(1.0)
    print('Switch on scalar filed solver!')
#######################################################
################# Only for scalar #####################
if LMTI == True and Restart == False:
    # rho = rho_l * torch.ones(input_shape, device=device)
    rho = torch.zeros(input_shape, device=device)
    rho = alpha*rho_l + (1-alpha)*rho_g*50
    print('Solving multiphase flows!')
else:
    print('Solving single-phase flows!')
################# Only for IBM ########################
if LIBM == True:
    mesh = np.load("Stokes.npy")
    sigma = torch.zeros(input_shape, dtype=torch.float32, device=device) 
    print(mesh.shape, sigma.shape)
    for i in range(nz):
        sigma[0,0,i,:,:] = torch.tensor(mesh[0,i+30,:,:,0])
        sigma[0,0,:,:,0:5] = 1000000000.0
        sigma[0,0,:,:,507:512] = 1000000000.0
        sigma[0,0,:,0:5,:] = 1000000000.0
        sigma[0,0,:,507:512,:] = 1000000000.0
    plt.imshow(sigma[0,0,:,:,256].cpu(),cmap='gray_r')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('AI4Stokes.jpg')
    plt.close()
#######################################################

# # # ################################### # # #
# # # #########  AI4MULTI MAIN ########## # # #
# # # ################################### # # #
class AI4MULTI(nn.Module):
    """docstring for AI4Multi"""
    def __init__(self):
        super(AI4MULTI, self).__init__()
        # self.arg = arg
        self.xadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.zadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.difx = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.dify = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.difz = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

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
        self.difx.weight.data = AD2_x
        self.dify.weight.data = AD2_y
        self.difz.weight.data = AD2_z

        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.zadv.bias.data = bias_initializer
        self.difx.bias.data = bias_initializer
        self.dify.bias.data = bias_initializer
        self.difz.bias.data = bias_initializer

###############################################################
    def boundary_condition_u(self, values_u, values_uu, nx, ny, nz, nnx, nny, nnz):
        values_uu[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_u[0,0,:,:,:]
        values_uu[0,0,:,:,0].fill_(0.0)
        values_uu[0,0,:,:,nx+1].fill_(0.0)
        values_uu[0,0,:,0,:] = values_uu[0,0,:,1,:]*0
        values_uu[0,0,:,ny+1,:] = values_uu[0,0,:,ny,:]*0
        values_uu[0,0,0,:,:] = values_uu[0,0,1,:,:]*0 
        values_uu[0,0,nz+1,:,:] = values_uu[0,0,nz,:,:]*0
        return values_uu

    def boundary_condition_v(self, values_v, values_vv, nx, ny, nz, nnx, nny, nnz):
        values_vv[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_v[0,0,:,:,:]
        values_vv[0,0,:,:,0] = values_vv[0,0,:,:,1]*0
        values_vv[0,0,:,:,nx+1] = values_vv[0,0,:,:,nx]*0
        values_vv[0,0,:,0,:].fill_(0.0)
        values_vv[0,0,:,ny+1,:].fill_(0.0)
        values_vv[0,0,0,:,:] = values_vv[0,0,1,:,:]*0 
        values_vv[0,0,nz+1,:,:] = values_vv[0,0,nz,:,:]*0
        return values_vv

    def boundary_condition_w(self, values_w, values_ww, nx, ny, nz, nnx, nny, nnz):
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

    def boundary_condition_pd(self, values_pd, values_pdd, nx, ny, nz, nnx, nny, nnz):
        values_pdd[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_pd[0,0,:,:,:]
        values_pdd[0,0,:,:,0] =  values_pdd[0,0,:,:,1]
        values_pdd[0,0,:,:,nx+1] = values_pdd[0,0,:,:,nx]
        values_pdd[0,0,:,0,:] = values_pdd[0,0,:,1,:]
        values_pdd[0,0,:,ny+1,:] = values_pdd[0,0,:,ny,:]
        values_pdd[0,0,0,:,:] = values_pdd[0,0,1,:,:]
        values_pdd[0,0,nz+1,:,:].fill_(0.0)
        return values_pdd

    def boundary_condition_ph(self, values_ph, values_phh, rho, nx, ny, nz, nnx, nny, nnz):
        values_phh[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_ph[0,0,:,:,:]
        values_phh[0,0,:,:,0] =  values_phh[0,0,:,:,1] 
        values_phh[0,0,:,:,nx+1] = values_phh[0,0,:,:,nx]
        values_phh[0,0,:,0,:] = values_phh[0,0,:,1,:]
        values_phh[0,0,:,ny+1,:] = values_phh[0,0,:,ny,:]
        values_phh[0,0,0,:,:] = values_phh[0,0,1,:,:] + dz * 10.0 * rho[0,0,1,:,:]
        values_phh[0,0,nz+1,:,:].fill_(0.0)
        return values_phh

    def boundary_condition_denstiy(self, values, valuesS, nx, ny, nz, nnx, nny, nnz):  # alpha, alphaa
        """ 
        values --> alpha, rho
        valuesS --> alphaa, rhoo
        """ 
        valuesS[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values[0,0,:,:,:]
        valuesS[0,0,:,:,0] = valuesS[0,0,:,:,1]
        valuesS[0,0,:,:,nx+1] = valuesS[0,0,:,:,nx]                 # test outflow boundary condition
# +++++++++++++++++++++++++++++++++++++++++++
        valuesS[0,0,:,0,:] = valuesS[0,0,:,1,:]
        valuesS[0,0,:,ny+1,:] = valuesS[0,0,:,ny,:]
        valuesS[0,0,0,:,:].fill_(rho_g)
        valuesS[0,0,nz+1,:,:] = valuesS[0,0,nz,:,:]
        return valuesS

    def boundary_condition_scalar(self, values, valuesS, nx, ny, nz, nnx, nny, nnz):  # alpha, alphaa
        """ 
        values --> alpha, rho
        valuesS --> alphaa, rhoo
        """ 
        valuesS[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values[0,0,:,:,:]
        valuesS[0,0,:,:,0] = valuesS[0,0,:,:,1]
        # valuesS[0,0,int(nz/2+1):nz+2,:,nx+1].fill_(0.0)             # test outflow boundary condition
        # valuesS[0,0,0:int(nz/2+1),:,nx+1].fill_(1.0)                # test outflow boundary condition
        valuesS[0,0,:,:,nx+1] = valuesS[0,0,:,:,nx]               # test outflow boundary condition
# +++++++++++++++++++++++++++++++++++++++++++
        valuesS[0,0,:,0,:] = valuesS[0,0,:,1,:]
        valuesS[0,0,:,ny+1,:] = valuesS[0,0,:,ny,:]
        valuesS[0,0,0,:,:].fill_(0.0)
        valuesS[0,0,nz+1,:,:] = valuesS[0,0,nz,:,:]
        return valuesS

    # def detect_scalar(self, values, valuesS, axis, nnx, nny, nnz):
    #     """ 
    #     values --> alpha, rho
    #     valuesS --> alphaa, rhoo
    #     axis: 4 --> x axis ; 3 --> y axis ; 2 --> z axis
    #     """  
    #     valuesS = (valuesS - torch.roll(valuesS, 1, axis)) * (torch.roll(valuesS, -1, axis) - valuesS)
    #     values[0,0,:,:,:] = valuesS[0,0,1:nnz-1,1:nny-1,1:nnx-1]
    #     return values
    def detect_scalar(self, values, axis, nnx, nny, nnz):
        """ 
        axis: 4 --> x axis ; 3 --> y axis ; 2 --> z axis
        """ 
        values = (values - torch.roll(values, 1, axis)) * (torch.roll(values, -1, axis) - values)
        # nz = values.shape[2]
        # ny = values.shape[3]
        # nx = values.shape[4]        
        if axis == 4:
              values[0,0,:,:,0].fill_(0.0)
              values[0,0,:,:,nx-1].fill_(0.0)
        elif axis == 3:
              values[0,0,:,0,:].fill_(0.0)
              values[0,0,:,ny-1,:].fill_(0.0)
        elif axis == 2:
              values[0,0,0,:,:].fill_(0.0)
              values[0,0,nz-1,:,:].fill_(0.0)
        return values

    def F_cycle_MG_pd(self, values_uu, values_vv, values_ww, rho, rho_old, rhoo, values_pd, values_pdd, iteration, diag, dt, nlevel, ratio, nx, ny, nz, nnx, nny, nnz):
        rhoo = self.boundary_condition_scalar(rho, rhoo, nx, ny, nz, nnx, nny, nnz)
        b = -(-self.xadv(values_uu * rhoo) - self.yadv(values_vv * rhoo) - self.zadv(values_ww * rhoo) - (rho - rho_old) / dt) / dt
        for MG in range(iteration):
            w = torch.zeros((1,1,1,1,1), device=device)
            r = self.A(self.boundary_condition_pd(values_pd, values_pdd, nx, ny, nz, nnx, nny, nnz)) - b 
            r_s = []  
            r_s.append(r)
            for i in range(1,nlevel-3):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel-3)):
                w = w - self.A(F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)) / diag + r_s[i] / diag
                w = self.prol(w)         
            values_pd = values_pd - w 
            values_pd = values_pd - self.A(self.boundary_condition_pd(values_pd, values_pdd, nx, ny, nz, nnx, nny, nnz)) / diag + b / diag
        return values_pd, w, r

    def F_cycle_MG_ph(self, values_ph, values_phh, rhoo, iteration, diag, nlevel, ratio, nx, ny, nz, nnx, nny, nnz):
        b = self.zadv(rhoo*int(abs(g_z)))
        for MG in range(iteration):  
            w = torch.zeros((1,1,1,1,1), device=device)
            r = self.A(self.boundary_condition_ph(values_ph, values_phh, rhoo, nx, ny, nz, nnx, nny, nnz)) - b 
            r_s = [] 
            r_s.append(r)
            for i in range(1,nlevel-3):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel-3)):
                w = w - self.A(F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)) / diag + r_s[i] / diag
                w = self.prol(w) 
            values_ph = values_ph - w
            values_ph = values_ph - self.A(self.boundary_condition_ph(values_ph, values_phh, rhoo, nx, ny, nz, nnx, nny, nnz)) / diag + b / diag
        return values_ph

    def PG_vector(self, values_uu, values_vv, values_ww, values_u, values_v, values_w, rho, k4, ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w): 
        resid = 1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz)
        k_u = 0.25 * dx * torch.abs(resid * AD2_u) / (1e-04  + (torch.abs(ADx_u) * dx**-3 + torch.abs(ADy_u) * dx**-3 + torch.abs(ADz_u) * dx**-3) / 3)
        k_v = 0.25 * dy * torch.abs(resid * AD2_v) / (1e-04  + (torch.abs(ADx_v) * dx**-3 + torch.abs(ADy_v) * dx**-3 + torch.abs(ADz_v) * dx**-3) / 3)
        k_w = 0.25 * dz * torch.abs(resid * AD2_w) / (1e-04  + (torch.abs(ADx_w) * dx**-3 + torch.abs(ADy_w) * dx**-3 + torch.abs(ADz_w) * dx**-3) / 3)

        k_u = torch.minimum(k_u, k4) * rho ; k_v = torch.minimum(k_v, k4) * rho ; k_w = torch.minimum(k_w, k4) * rho

        k_uu = F.pad(k_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(k_v, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_ww = F.pad(k_w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        k_x = 0.5 * (k_u * AD2_u + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * AD2_v + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        k_z = 0.5 * (k_w * AD2_w + self.diff(values_ww * k_ww) - values_w * self.diff(k_ww))
        return k_x, k_y, k_z

    def PG_compressive_scalar(self, alphaa, alpha, values_u, values_v, values_w, k1, k2, k3, k5, k6, ADx_c, ADy_c, ADz_c, nnx, nny, nnz):
        factor_S = 10  # Negative factor to be mutiplied by S (detecting variable)
        factor_P = 10 # Postive factor to be mutiplied by S (detecting variable)
        factor_beta = 0.1        
        
        temp1 = ADx_c ; temp2 = ADy_c ; temp3 = ADz_c

        temp4 = temp1 * (values_u * temp1 + values_v * temp2 + values_w * temp3) / (eplsion_k + temp1**2 + temp2**2 + temp3**2)
        temp5 = temp2 * (values_u * temp1 + values_v * temp2 + values_w * temp3) / (eplsion_k + temp1**2 + temp2**2 + temp3**2)
        temp6 = temp3 * (values_u * temp1 + values_v * temp2 + values_w * temp3) / (eplsion_k + temp1**2 + temp2**2 + temp3**2)

        mag3 = (eplsion_k + (torch.abs(temp1 * (det**-1)) + torch.abs(temp2 * (det**-1)) + torch.abs(temp3 * (det**-1))) / 3)  
        mag4 = 1/3 * (det**-1) * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz)

        Jug = torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0))
        detectx = self.detect_scalar(alpha,4,nnx,nny,nnz) ; detecty = self.detect_scalar(alpha,3,nnx,nny,nnz) ; detectz = self.detect_scalar(alpha,2,nnx,nny,nnz)

        k_u = torch.minimum(k1, torch.maximum(k2, factor_S * torch.where(Jug, k3, detectx ))) * -factor_beta * values_u**2 / (eplsion_k + temp1**2 * torch.abs(temp4)) / dx + \
             -torch.maximum(k3, torch.minimum(k2, factor_P * torch.where(Jug, k3, detectx ))) * 3 * dx * torch.abs(mag4 * self.diff(alphaa)) / mag3

        k_v = torch.minimum(k1, torch.maximum(k2, factor_S * torch.where(Jug, k3, detecty ))) * -factor_beta * values_v**2 / (eplsion_k + temp2**2 * torch.abs(temp5)) / dy + \
             -torch.maximum(k3, torch.minimum(k2, factor_P * torch.where(Jug, k3, detecty ))) * 3 * dx * torch.abs(mag4 * self.diff(alphaa)) / mag3

        k_w = torch.minimum(k1, torch.maximum(k2, factor_S * torch.where(Jug, k3, detectz ))) * -factor_beta * values_w**2 / (eplsion_k + temp3**2 * torch.abs(temp6)) / dz + \
             -torch.maximum(k3, torch.minimum(k2, factor_P * torch.where(Jug, k3, detectz ))) * 3 * dx * torch.abs(mag4 * self.diff(alphaa)) / mag3

        k_u = torch.where(torch.gt(k_u,0.0),torch.minimum(k_u,k5),torch.maximum(k_u,k6))
        k_v = torch.where(torch.gt(k_v,0.0),torch.minimum(k_v,k5),torch.maximum(k_v,k6))
        k_w = torch.where(torch.gt(k_w,0.0),torch.minimum(k_w,k5),torch.maximum(k_w,k6))

        k_uu = F.pad(k_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(k_v, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_ww = F.pad(k_w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        k_x = 0.5 * (k_u * self.difx(alphaa) + self.difx(alphaa * k_uu) - alpha * self.difx(k_uu)) + \
              0.5 * (k_v * self.dify(alphaa) + self.dify(alphaa * k_vv) - alpha * self.dify(k_vv)) + \
              0.5 * (k_w * self.difz(alphaa) + self.difz(alphaa * k_ww) - alpha * self.difz(k_ww))  
        return k_x

    # **************************** surface tensor implementation ****************************
    # def surface_tension(self, alpha, alphaa):
    #     alphaa = self.boundary_condition_scalar(alpha, alphaa, nx, ny, nz, nnx, nny, nnz)
    #     norm_x = self.xadv(alphaa) / ((torch.abs(self.xadv(alphaa))**2 + torch.abs(self.yadv(alphaa))**2 + torch.abs(self.zadv(alphaa))**2)**0.5 + 1e-04)
    #     norm_y = self.yadv(alphaa) / ((torch.abs(self.xadv(alphaa))**2 + torch.abs(self.yadv(alphaa))**2 + torch.abs(self.zadv(alphaa))**2)**0.5 + 1e-04)
    #     norm_z = self.zadv(alphaa) / ((torch.abs(self.xadv(alphaa))**2 + torch.abs(self.yadv(alphaa))**2 + torch.abs(self.zadv(alphaa))**2)**0.5 + 1e-04)

    #     norm_xx = F.pad(norm_x, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    #     norm_yy = F.pad(norm_y, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    #     norm_zz = F.pad(norm_z, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        
    #     k_x = self.xadv(norm_xx) + self.yadv(norm_yy) + self.zadv(norm_zz)
 
    #     F_x = 2.0*1*5*k_x*self.xadv(alphaa)
    #     F_y = 2.0*1*5*k_x*self.yadv(alphaa)
    #     F_z = 2.0*1*5*k_x*self.zadv(alphaa)
    #     return F_x, F_y, F_z
    # **************************** surface tensor implementation ****************************

    def forward(self, values_u, values_uu, values_v, values_vv, values_w, values_ww, values_pd, values_pdd, values_ph, values_phh, alpha, alphaa, rho, rhoo, b_uu, b_vv, b_ww, k1, k2, k3, k4, k5, k6, k7, k8, dt, rho_l, rho_g, iteration):
    # **************************** surface tensor implementation ****************************
    # [F_x,F_y,F_z] = self.surface_tension(alpha,alphaa)
    # **************************** surface tensor implementation ****************************
    # Hydrostatic pressure 
        rhoo = self.boundary_condition_scalar(rho, rhoo, nx, ny, nz, nnx, nny, nnz)
        values_ph = self.F_cycle_MG_ph(values_ph, values_phh, rhoo, iteration, diag, nlevel, ratio, nx, ny, nz, nnx, nny, nnz)
    # Padding velocity vectors 
        values_uu = self.boundary_condition_u(values_u, values_uu, nx, ny, nz, nnx, nny, nnz)
        values_vv = self.boundary_condition_v(values_v, values_vv, nx, ny, nz, nnx, nny, nnz)
        values_ww = self.boundary_condition_w(values_w, values_ww, nx, ny, nz, nnx, nny, nnz)
        values_phh = self.boundary_condition_ph(values_ph, values_phh, rhoo, nx, ny, nz, nnx, nny, nnz)

        Grapx_p = self.xadv(values_phh) / rho * dt ; Grapy_p = self.yadv(values_phh) / rho * dt ; Grapz_p = self.zadv(values_phh) / rho * dt  
        ADx_u = self.xadv(values_uu) ; ADy_u = self.yadv(values_uu) ; ADz_u = self.zadv(values_uu)
        ADx_v = self.xadv(values_vv) ; ADy_v = self.yadv(values_vv) ; ADz_v = self.zadv(values_vv)
        ADx_w = self.xadv(values_ww) ; ADy_w = self.yadv(values_ww) ; ADz_w = self.zadv(values_ww)
        AD2_u = self.diff(values_uu) ; AD2_v = self.diff(values_vv) ; AD2_w = self.diff(values_ww)

        [k_x,k_y,k_z] = self.PG_vector(values_uu, values_vv, values_ww, values_u, values_v, values_w, rho, k4, ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w)
    # First step for solving uvw
        b_u = values_u + 0.5 * (k_x * dt / rho - values_u * ADx_u * dt - values_v * ADy_u * dt - values_w * ADz_u * dt) - Grapx_p             #  - F_x * dt / rho ******************************
        b_v = values_v + 0.5 * (k_y * dt / rho - values_u * ADx_v * dt - values_v * ADy_v * dt - values_w * ADz_v * dt) - Grapy_p             #  - F_x * dt / rho surface tensor implementation
        b_w = values_w + 0.5 * (k_z * dt / rho - values_u * ADx_w * dt - values_v * ADy_w * dt - values_w * ADz_w * dt) + g_z * dt - Grapz_p  #  - F_x * dt / rho ******************************
    # Solid body
        if LIBM == True: [b_u, b_v, b_w] = self.solid_body(b_u, b_v, b_w, sigma, dt)
    # Padding velocity vectors 
        b_uu = self.boundary_condition_u(b_u, b_uu, nx, ny, nz, nnx, nny, nnz)
        b_vv = self.boundary_condition_v(b_v, b_vv, nx, ny, nz, nnx, nny, nnz)
        b_ww = self.boundary_condition_w(b_w, b_ww, nx, ny, nz, nnx, nny, nnz)

        ADx_u = self.xadv(b_uu) ; ADy_u = self.yadv(b_uu) ; ADz_u = self.zadv(b_uu)
        ADx_v = self.xadv(b_vv) ; ADy_v = self.yadv(b_vv) ; ADz_v = self.zadv(b_vv)
        ADx_w = self.xadv(b_ww) ; ADy_w = self.yadv(b_ww) ; ADz_w = self.zadv(b_ww)
        AD2_u = self.diff(b_uu) ; AD2_v = self.diff(b_vv) ; AD2_w = self.diff(b_ww)

        [k_x,k_y,k_z] = self.PG_vector(b_uu, b_vv, b_ww, b_u, b_v, b_w, rho, k4, ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w)
    # Second step for solving uvw   
        values_u = values_u + k_x * dt / rho - b_u * ADx_u * dt - b_v * ADy_u * dt - b_w * ADz_u * dt - Grapx_p             #  - F_x * dt / rho ******************************
        values_v = values_v + k_y * dt / rho - b_u * ADx_v * dt - b_v * ADy_v * dt - b_w * ADz_v * dt - Grapy_p             #  - F_x * dt / rho surface tensor implementation
        values_w = values_w + k_z * dt / rho - b_u * ADx_w * dt - b_v * ADy_w * dt - b_w * ADz_w * dt + g_z * dt - Grapz_p  #  - F_x * dt / rho ******************************
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
    # Transport indicator field 
        alphaa = self.boundary_condition_scalar(alpha, alphaa, nx, ny, nz, nnx, nny, nnz)
        ADx_c = self.xadv(alphaa) ; ADy_c = self.yadv(alphaa) ; ADz_c = self.zadv(alphaa)
        b_u = alpha + 0.5 * (self.PG_compressive_scalar(alphaa, alpha, values_u, values_v, values_w, k1, k2, k3, k5, k6, ADx_c, ADy_c, ADz_c, nnx, nny, nnz) * dt - \
        values_u * ADx_c * dt - values_v * ADy_c * dt - values_w * ADz_c * dt)
    # 
        b_uu = self.boundary_condition_scalar(b_u, b_uu, nx, ny, nz, nnx, nny, nnz)
        b_u = torch.maximum(torch.minimum(b_u,k1),k2)
        b_uu = torch.maximum(torch.minimum(self.boundary_condition_scalar(b_u, b_uu, nx, ny, nz, nnx, nny, nnz),k7),k8)
        ADx_c = self.xadv(b_uu) ; ADy_c = self.yadv(b_uu) ; ADz_c = self.zadv(b_uu)
    # 
        alpha = alpha + (self.PG_compressive_scalar(b_uu, b_u, values_u, values_v, values_w, k1, k2, k3, k5, k6, ADx_c, ADy_c, ADz_c, nnx, nny, nnz) * dt - \
        values_u * ADx_c * dt - values_v * ADy_c * dt - values_w * ADz_c * dt) 
    # Avoid sharp interfacing    
        alpha = torch.minimum(alpha,torch.ones(input_shape, device=device))
        alpha = torch.maximum(alpha,torch.zeros(input_shape, device=device))
        rho_old = rho
        rho = alpha*rho_l + (1-alpha) * rho_g * 50
    # non-hydrostatic pressure
        rhoo = self.boundary_condition_scalar(rho, rhoo, nx, ny, nz, nnx, nny, nnz)
        values_uu = self.boundary_condition_u(values_u, values_uu, nx, ny, nz, nnx, nny, nnz)
        values_vv = self.boundary_condition_v(values_v, values_vv, nx, ny, nz, nnx, nny, nnz)
        values_ww = self.boundary_condition_w(values_w, values_ww, nx, ny, nz, nnx, nny, nnz)   
        [values_pd, w ,r] = self.F_cycle_MG_pd(values_uu, values_vv, values_ww, rho, rho_old, rhoo, values_pd, values_pdd, iteration, diag, dt, nlevel, ratio, nx, ny, nz, nnx, nny, nnz)
    # Pressure gradient correction - non-hydrostatic     
        values_pdd = self.boundary_condition_pd(values_pd, values_pdd, nx, ny, nz, nnx, nny, nnz)
        values_u = values_u + self.xadv(values_pdd) / rho * dt 
        values_v = values_v + self.yadv(values_pdd) / rho * dt 
        values_w = values_w + self.zadv(values_pdd) / rho * dt   
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
        return values_u, values_v, values_w, values_ph, values_pd, alpha, rho, w, r

model = AI4MULTI().to(device)

start = time.time()
with torch.no_grad():
    for itime in range(1,ntime+1):
       ###############################################################
       [values_u, values_v, values_w, values_ph, values_pd, alpha, rho, w, r] = model(values_u, values_uu, values_v, values_vv, values_w, values_ww,
                     values_pd, values_pdd, values_ph, values_phh, alpha, alphaa, rho, rhoo, b_uu, b_vv, b_ww, k1, k2, k3, k4, k5, k6, k7, k8, dt, rho_l, rho_g, iteration)
# output  
       real_time = real_time + dt
       istep +=1 
       print('Time step:', itime) 
       print('Pressure error:', np.max(np.abs(w.cpu().detach().numpy())), 'cty equation residual:', np.max(np.abs(alpha.cpu().detach().numpy())))
       print('========================================================')
       if np.max(np.abs(w.cpu().detach().numpy())) > 80000.0:
              np.save(save_path+"/dbug_alpha"+str(itime), arr=alpha.cpu().detach().numpy()[0,0,:,:])              
              np.save(save_path+"/dbug_w"+str(itime), arr=values_w.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/dbug_v"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/dbug_u"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])
              print('Not converged !!!!!!')
              break
       if save_fig == True and itime % n_out == 0:
              np.save(save_path+"/alpha"+str(itime), arr=alpha.cpu().detach().numpy()[0,0,:,:])              
              np.save(save_path+"/w"+str(itime), arr=values_w.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/v"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/u"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])
       if itime == ntime:
              np.save(save_path+"/alpha"+str(itime), arr=alpha.cpu().detach().numpy()[0,0,:,:])          
              np.save(save_path+"/rho"+str(itime), arr=rho.cpu().detach().numpy()[0,0,:,:])                  
              np.save(save_path+"/w"+str(itime), arr=values_w.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/v"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/u"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/pd"+str(itime), arr=values_ph.cpu().detach().numpy()[0,0,:,:])
              np.save(save_path+"/ph"+str(itime), arr=values_pd.cpu().detach().numpy()[0,0,:,:])
    end = time.time()
    print('time',(end-start))
