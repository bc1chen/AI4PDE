import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import AI4CFD_filters as CNN3D

# # # ######################################### # # #
# # # ###### Functions linking to the AI ###### # # #
# # # ######################################### # # #
def boundary_condition_velocityU(values_u, nx, ny, nz):
    'Define boundary conditions for velocity field'
    '''
    values_u -> velocity in x direction
    nx,ny,nz -> grid point in each direction
    '''
    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension

    # Pad values_u, values_v, and values_w
    values_u = F.pad(values_u, paddings, mode='constant', value=0)

    values_u[0,0,:,:,0].fill_(0.0)
    values_u[0,0,:,:,nx+1].fill_(0.0)

    values_u[0,0,:,0,:]=values_u[0,0,:,1,:]
    values_u[0,0,:,ny+1,:]=values_u[0,0,:,ny,:]

    values_u[0,0,0,:,:]=values_u[0,0,1,:,:]
    values_u[0,0,nz+1,:,:]=values_u[0,0,nz,:,:]

    return values_u

def boundary_condition_velocityV(values_v, nx, ny, nz):
    'Define boundary conditions for velocity field'
    '''
    values_v -> velocity in y direction
    nx,ny,nz -> grid point in each direction
    '''
    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension
    # Pad values_u, values_v, and values_w
    values_v = F.pad(values_v, paddings, mode='constant', value=0)

    values_v[0,0,:,:,0] = values_v[0,0,:,:,1]
    values_v[0,0,:,:,nx+1] = values_v[0,0,:,:,nx]

    values_v[0,0,:,0,:].fill_(0.0)
    values_v[0,0,:,ny+1,:].fill_(0.0)

    values_v[0,0,0,:,:]=values_v[0,0,1,:,:]
    values_v[0,0,nz+1,:,:]=values_v[0,0,nz,:,:]

    return values_v

def boundary_condition_velocityW(values_w, nx, ny, nz):
    'Define boundary conditions for velocity field'
    '''
    values_w -> velocity in z direction
    nx,ny,nz -> grid point in each direction
    '''
    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension

    # Pad values_u, values_v, and values_w
    values_w = F.pad(values_w, paddings, mode='constant', value=0)

    values_w[0,0,:,:,0] = values_w[0,0,:,:,1]
    values_w[0,0,:,:,nx+1] = values_w[0,0,:,:,nx]

    values_w[0,0,:,0,:]=values_w[0,0,:,1,:]
    values_w[0,0,:,ny+1,:]=values_w[0,0,:,ny,:]

    values_w[0,0,0,:,:].fill_(0.0)
    values_w[0,0,nz+1,:,:].fill_(0.0)

    return values_w


def boundary_condition_pressure_d(values_p, nx, ny, nz):
    'Define boundary conditions for non-hydrostatic (dynamic) pressure field'
    '''
    values_p -> non-hydrostatic pressure
    nx,ny,nz -> grid point in each direction
    '''
    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension
    values_p = F.pad(values_p, paddings, mode='constant', value=0)

    values_p[0,0,:,:,0]=values_p[0,0,:,:,1]
    values_p[0,0,:,:,nx+1]=values_p[0,0,:,:,nx]

    values_p[0,0,:,0,:]=values_p[0,0,:,1,:]
    values_p[0,0,:,ny+1,:]=values_p[0,0,:,ny,:]
    
    values_p[0,0,0,:,:]=values_p[0,0,1,:,:]
    values_p[0,0,nz+1,:,:].fill_(0.0)
    #this should be same with below assignment
    #temp1[0,nz+1,:,:,0].assign(tf.Variable(tf.zeros((1,ny+2,nx+2)))[0,:]) 
    return values_p

def boundary_condition_pressure_h(values_p, nx, ny, nz, rho, dz):
    'Define boundary conditions for hydrostatic pressure field'
    '''
    values_p -> hydrostatic pressure
    nx,ny,nz -> grid point in each direction
    rho -> density 
    dz -> grid spacing in z direction (gravity)
    '''
    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension
    values_p = F.pad(values_p, paddings, mode='constant', value=0)

    values_p[0,0,:,:,0]=values_p[0,0,:,:,1]
    values_p[0,0,:,:,nx+1]=values_p[0,0,:,:,nx]
    
    values_p[0,0,:,0,:]=values_p[0,0,:,1,:]
    values_p[0,0,:,ny+1,:]=values_p[0,0,:,ny,:]
    
    values_p[0,0,0,:,:]=values_p[0,0,1,:,:]+dz*10.0*rho[0,0,1,:,:]
    values_p[0,0,nz+1,:,:].fill_(0.0)
    
    return values_p

def boundary_condition_density(rho, nx, ny, nz):
    'Define boundary conditions for pressure field'
    '''
    rho -> density 
    nx,ny,nz -> grid point in each direction
    '''
    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension
    rho = F.pad(rho, paddings, mode='constant', value=0)

    rho[0,0,:,:,0]=rho[0,0,:,:,1]
    rho[0,0,:,:,nx+1]=rho[0,0,:,:,nx]
    
    rho[0,0,:,0,:]=rho[0,0,:,1,:]
    rho[0,0,:,ny+1,:]=rho[0,0,:,ny,:]

    rho[0,0,0,:,:]=rho[0,0,1,:,:]
    rho[0,0,nz+1,:,:]=rho[0,0,nz,:,:]

    return rho

def boundary_condition_indicator_SAME(alpha, nx, ny, nz):   
    'Define boundary conditions for scalar field'
    '''
    alpha -> indicator  
    nx,ny,nz -> grid point in each direction
    '''

    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension
    alpha = F.pad(alpha, paddings, mode='constant', value=0)

    
    alpha[0,0,:,:,0]=alpha[0,0,:,:,1]
    alpha[0,0,:,:,nx+1]=alpha[0,0,:,:,nx]

    alpha[0,0,:,0,:]=alpha[0,0,:,1,:]
    alpha[0,0,:,ny+1,:]=alpha[0,0,:,ny,:]

    alpha[0,0,0,:,:]=alpha[0,0,1,:,:]
    alpha[0,0,nz+1,:,:]=alpha[0,0,nz,:,:]

    return alpha

def boundary_condition_coeff_SAME(k_x, k_y, k_z, nx, ny, nz):   
    'Define boundary conditions for scalar field'
    '''
    k_x -> diffusion coefficent in x direction
    k_y -> diffusion coefficent in y direction
    k_z -> diffusion coefficent in z direction
    nx,ny,nz -> grid point in each direction
    '''
    paddings = (1, 1, 1, 1, 1 ,1)  # Padding amounts for each dimension
    k_x= F.pad(k_x, paddings, mode='constant', value=0)
    k_y= F.pad(k_y, paddings, mode='constant', value=0)
    k_z= F.pad(k_z, paddings, mode='constant', value=0)

    k_x[0,0,:,:,0].fill_(0.0)
    k_y[0,0,:,:,0].fill_(0.0)
    k_z[0,0,:,:,0].fill_(0.0)

    k_x[0,0,:,:,nx+1].fill_(0.0)
    k_y[0,0,:,:,nx+1].fill_(0.0)
    k_z[0,0,:,:,nx+1].fill_(0.0)

    k_x[0,0,:,0,:].fill_(0.0)
    k_y[0,0,:,0,:].fill_(0.0)
    k_z[0,0,:,0,:].fill_(0.0)
   
    k_x[0,0,:,ny+1,:].fill_(0.0)
    k_y[0,0,:,ny+1,:].fill_(0.0)
    k_z[0,0,:,ny+1,:].fill_(0.0)
       
    k_x[0,0,0,:,:].fill_(0.0)
    k_y[0,0,0,:,:].fill_(0.0)
    k_z[0,0,0,:,:].fill_(0.0)
 
    k_x[0,0,nz+1,:,:].fill_(0.0)
    k_y[0,0,nz+1,:,:].fill_(0.0)
    k_z[0,0,nz+1,:,:].fill_(0.0)
    return k_x, k_y, k_z

def save_data(alpha,rho,n_out,itime):
    'Save field data'
    if itime % n_out == 0:  
        np.save("Results/Water_Column_3D_512/rho"+str(itime), arr=rho[0,0,:,:,:])
        np.save("Results/Water_Column_3D_512/C"+str(itime), arr=alpha[0,0,:,:,:])

# # # ################################### # # #
# # # ######    Petrov Galerkin    ###### # # #
# # # ################################### # # #
def PG_vector_SAME(values_uu, values_vv, values_ww, values_u, values_v, values_w, eplsion_k, rho, nx, ny, nz, dx, dy, dz, dt):    
    'Turbulence modelling using Petrov-Galerkin dissipation'     
    '''  
    ---------------------------------------------------------------------
    Input
    ---------------------------------------------------------------------
    values_uu: padded u-component velocity 
    values_vv: padded v-component velocity 
    values_ww: padded w-component velocity 
    values_u: u-component velocity 
    values_v: v-component velocity 
    values_w: w-component velocity 
    eplsion_k: Need to sufficiently large
    rho: density 
    nx,ny,nz: grid point in each direction
    dx,dy,dz: grid spacing in each direction
    dt: time step
    ---------------------------------------------------------------------
    Output
    ---------------------------------------------------------------------
    temp1: Final diffusion matrix in x direction 
    temp2: Final diffusion matrix in y direction 
    temp3: Final diffusion matrix in z direction 
    '''
    input_shape = (1,1,nz,ny,nx)
# x direction 
    temp1 = 0.25 * dx * abs(1/3*(dx**-3)*
                               (abs(values_u)*dx + abs(values_v)*dx + abs(values_w)*dx) * 
                               CNN3D.dif(values_uu)) / (eplsion_k + 
                               (abs(CNN3D.xadv(values_uu))*(dx**-3) + 
                               abs(CNN3D.yadv(values_uu))*(dx**-3) +
                               abs(CNN3D.zadv(values_uu))*(dx**-3))/3)
# y direction  
    temp2 = 0.25 * dy * abs(1/3*(dx**-3)*
                               (abs(values_u)*dy + abs(values_v)*dy + abs(values_w)*dy) * 
                               CNN3D.dif(values_vv)) / (eplsion_k + 
                               (abs(CNN3D.xadv(values_vv))*(dy**-3) + 
                               abs(CNN3D.yadv(values_vv))*(dy**-3) +
                               abs(CNN3D.zadv(values_vv))*(dy**-3))/3)
# z direction  
    temp3 = 0.25 * dz * abs(1/3*(dx**-3)*
                               (abs(values_u)*dz + abs(values_v)*dz + abs(values_w)*dz) * 
                               CNN3D.dif(values_ww)) / (eplsion_k + 
                               (abs(CNN3D.xadv(values_ww))*(dz**-3) + 
                               abs(CNN3D.yadv(values_ww))*(dz**-3) + 
                               abs(CNN3D.zadv(values_ww))*(dz**-3))/3)

    temp1 = rho*torch.minimum(temp1, torch.ones(input_shape)*dx**2*0.25/dt) 
    temp2 = rho*torch.minimum(temp2, torch.ones(input_shape)*dy**2*0.25/dt)  
    temp3 = rho*torch.minimum(temp3, torch.ones(input_shape)*dz**2*0.25/dt)  
# x direction     
    temp4 = 0.5 * (temp1 * CNN3D.dif(values_uu) +
        CNN3D.dif(values_uu * boundary_condition_coeff_SAME(temp1,temp2,temp3,nx,ny,nz)[0]) -
        values_u * CNN3D.dif(boundary_condition_coeff_SAME(temp1,temp2,temp3,nx,ny,nz)[0]))
# y direction 
    temp5 = 0.5 * (temp2 * CNN3D.dif(values_vv) + 
        CNN3D.dif(values_vv * boundary_condition_coeff_SAME(temp1,temp2,temp3,nx,ny,nz)[1]) - 
        values_v * CNN3D.dif(boundary_condition_coeff_SAME(temp1,temp2,temp3,nx,ny,nz)[1]))
# z direction 
    temp6 = 0.5 * (temp3 * CNN3D.dif(values_ww) + 
        CNN3D.dif(values_ww * boundary_condition_coeff_SAME(temp1,temp2,temp3,nx,ny,nz)[2]) - 
        values_w * CNN3D.dif(boundary_condition_coeff_SAME(temp1,temp2,temp3,nx,ny,nz)[2]))
    return temp4, temp5, temp6

def PG_turb_scalar_SAME(alphaa, alpha, values_u, values_v, values_w, eplsion_k, nx, ny, nz, dx, dt):    
    '''Turbulence modelling using Petrov-Galerkin dissipation       
    Input
    ---------------------------------------------------------------------
    alphaa: padded indicator
    alpha: indicator  
    values_u: u-component velocity 
    values_v: v-component velocity 
    values_w: w-component velocity 
    eplsion_k: Need to sufficiently large
    Output
    ---------------------------------------------------------------------
    k_alpha (temp10): Final diffusion matrix in x direction 
    '''
    input_shape = (1,1,nz,ny,nx)
    factor_S = 1 # Negative factor to be mutiplied by S (detecting variable)
    factor_P = 1 # Postive factor to be mutiplied by S (detecting variable)
    factor_beta = 0.1

    temp1 = CNN3D.xadv(alphaa) 
    temp2 = CNN3D.yadv(alphaa)  
    temp3 = CNN3D.zadv(alphaa)

    temp4 = temp1*(values_u*temp1+values_v*temp2+values_w*temp3)/\
    (eplsion_k+temp1**2+temp2**2+temp3**2)
    temp5 = temp2*(values_u*temp1+values_v*temp2+values_w*temp3)/\
    (eplsion_k+temp1**2+temp2**2+temp3**2)
    temp6 = temp3*(values_u*temp1+values_v*temp2+values_w*temp3)/\
    (eplsion_k+temp1**2+temp2**2+temp3**2)
# Diffusion coefficient in x direction  
    temp7 = torch.minimum(torch.ones(input_shape) * 1.0, torch.maximum(torch.zeros(input_shape),
        factor_S * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), torch.ones(input_shape) * -1.0,
            CNN3D.wxu(alphaa) * CNN3D.wxd(alphaa)))) * \
    -factor_beta * (values_u**2 + values_v**2 + values_w**2) / (eplsion_k + (temp1**2 + temp2**2 + temp3**2) * \
        (abs(temp4) + abs(temp5) + abs(temp6))/3)/dx + \
    -torch.maximum(torch.ones(input_shape) * -1.0, torch.minimum(torch.zeros(input_shape),        # negative (above) and postive (below)
        factor_P * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), torch.ones(input_shape) * -1.0,
            CNN3D.wxu(alphaa) * CNN3D.wxd(alphaa)))) * \
    3 * dx * abs(1/3 * (dx**-3) * (abs(values_u) * dx + abs(values_v) * dx + abs(values_w) * dx) * \
        CNN3D.dif(alphaa)) / (eplsion_k + (abs(temp1 * (dx**-3)) + abs(temp2 * (dx**-3)) + abs(temp3 * (dx**-3))) / 3)  
# Diffusion coefficient in y direction  
    temp8 = torch.minimum(torch.ones(input_shape) * 1.0, torch.maximum(torch.zeros(input_shape),
        factor_S * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), torch.ones(input_shape) * -1.0,
            CNN3D.wyu(alphaa) * CNN3D.wyd(alphaa)))) * \
    -factor_beta * (values_u**2 + values_v**2 + values_w**2) / (eplsion_k + (temp1**2 + temp2**2 + temp3**2) * \
        (abs(temp4) + abs(temp5) + abs(temp6))/3)/dx + \
    -torch.maximum(torch.ones(input_shape) * -1.0, torch.minimum(torch.zeros(input_shape),        # negative (above) and postive (below)
        factor_P * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), torch.ones(input_shape) * -1.0,
            CNN3D.wyu(alphaa) * CNN3D.wyd(alphaa)))) * \
    3 * dx * abs(1/3 * (dx**-3) * (abs(values_u) * dx + abs(values_v) * dx + abs(values_w) * dx) * \
        CNN3D.dif(alphaa)) / (eplsion_k + (abs(temp1 * (dx**-3)) + abs(temp2 * (dx**-3)) + abs(temp3 * (dx**-3))) / 3)  
# Diffusion coefficient in z direction  
    temp9 = torch.minimum(torch.ones(input_shape) * 1.0, torch.maximum(torch.zeros(input_shape),  
        factor_S * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), torch.ones(input_shape) * -1.0,
            CNN3D.wzu(alphaa) * CNN3D.wzd(alphaa)))) * \
    -factor_beta * (values_u**2 + values_v**2 + values_w**2) / (eplsion_k + (temp1**2 + temp2**2 + temp3**2) * \
        (abs(temp4) + abs(temp5) + abs(temp6))/3)/dx + \
    -torch.maximum(torch.ones(input_shape) * -1.0, torch.minimum(torch.zeros(input_shape),        # negative (above) and postive (below)
        factor_P * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), torch.ones(input_shape) * -1.0,
            CNN3D.wzu(alphaa) * CNN3D.wzd(alphaa)))) * \
    3 * dx * abs(1/3 * (dx**-3) * (abs(values_u) * dx + abs(values_v) * dx + abs(values_w) * dx) * \
        CNN3D.dif(alphaa)) / (eplsion_k + (abs(temp1 * (dx**-3)) + abs(temp2 * (dx**-3)) + abs(temp3 * (dx**-3))) / 3)  

    temp7 = torch.where(torch.gt(temp7,0.0),torch.minimum(temp7,torch.ones(input_shape)*dx**2*0.05/dt),torch.maximum(temp7,torch.ones(input_shape)*dx**2*-0.0001/dt))
    temp8 = torch.where(torch.gt(temp8,0.0),torch.minimum(temp8,torch.ones(input_shape)*dx**2*0.05/dt),torch.maximum(temp8,torch.ones(input_shape)*dx**2*-0.0001/dt))
    temp9 = torch.where(torch.gt(temp9,0.0),torch.minimum(temp9,torch.ones(input_shape)*dx**2*0.05/dt),torch.maximum(temp9,torch.ones(input_shape)*dx**2*-0.0001/dt))
        
    temp10 = 0.5 * (temp7 * CNN3D.difx(alphaa) + CNN3D.difx(alphaa * boundary_condition_coeff_SAME(temp7,temp8,temp9,nx,ny,nz)[0]) - \
        alpha * CNN3D.difx(boundary_condition_coeff_SAME(temp7,temp8,temp9,nx,ny,nz)[0])) + \
    0.5 * (temp8 * CNN3D.dify(alphaa) + CNN3D.dify(alphaa * boundary_condition_coeff_SAME(temp7,temp8,temp9,nx,ny,nz)[1]) - \
        alpha * CNN3D.dify(boundary_condition_coeff_SAME(temp7,temp8,temp9,nx,ny,nz)[1])) + \
    0.5 * (temp9 * CNN3D.difz(alphaa) + CNN3D.difz(alphaa * boundary_condition_coeff_SAME(temp7,temp8,temp9,nx,ny,nz)[2]) - \
        alpha * CNN3D.difz(boundary_condition_coeff_SAME(temp7,temp8,temp9,nx,ny,nz)[2]))    
    return temp10