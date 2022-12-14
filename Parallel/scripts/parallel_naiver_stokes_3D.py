# TensorFlow ≥2.0 is required
from halo_exchange_upgraded import HaloExchange
from time import perf_counter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # avoid tf dnn flag issue
import numpy as np
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# CFD Parameters
dt = 0.1        # time step (s)
dx = 1          # grid size (m)
Re = 1/4        # diffusion coefficient (m.s-2)
ub = 1          # bulk velocity (m/s)
sigma = 100000  # Absorption coefficent for buildings
nx = 128        # Grid point in x
ny = 128        # Grid point in y
nz = 128        # Grid point in z

# Weights of CNNs layers
pd1 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
pd2 = [[3/26, 6/26,  3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26,  3/26]]
pd3 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]

w1 = np.zeros([1, 3, 3, 3, 1])
w1[0, 0, :, :, 0] = np.array(pd1)*dt*Re/dx**2
w1[0, 1, :, :, 0] = np.array(pd2)*dt*Re/dx**2
w1[0, 2, :, :, 0] = np.array(pd3)*dt*Re/dx**2

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

w2 = np.zeros([1, 3, 3, 3, 1])
w3 = np.zeros([1, 3, 3, 3, 1])
w4 = np.zeros([1, 3, 3, 3, 1])

w2[0, 0, :, :, 0] = np.array(p_div_x1)*dt/dx
w2[0, 1, :, :, 0] = np.array(p_div_x2)*dt/dx
w2[0, 2, :, :, 0] = np.array(p_div_x3)*dt/dx

w3[0, 0, :, :, 0] = np.array(p_div_y1)*dt/dx
w3[0, 1, :, :, 0] = np.array(p_div_y2)*dt/dx
w3[0, 2, :, :, 0] = np.array(p_div_y3)*dt/dx

w4[0, 0, :, :, 0] = np.array(p_div_z1)*dt/dx
w4[0, 1, :, :, 0] = np.array(p_div_z2)*dt/dx
w4[0, 2, :, :, 0] = np.array(p_div_z3)*dt/dx

pA1 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
pA2 = [[3/26, 6/26,  3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26,  3/26]]
pA3 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]

w5 = np.zeros([1, 3, 3, 3, 1])
w5[0, 0, :, :, 0] = -np.array(pA1)/dx**2
w5[0, 1, :, :, 0] = -np.array(pA2)/dx**2
w5[0, 2, :, :, 0] = -np.array(pA3)/dx**2

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

w6 = np.zeros([1, 3, 3, 3, 1])
w7 = np.zeros([1, 3, 3, 3, 1])
w8 = np.zeros([1, 3, 3, 3, 1])
w9 = np.zeros([1, 2, 2, 2, 1])
w6[0, 0, :, :, 0] = np.array(pctyu1)/(dx*dt)
w6[0, 1, :, :, 0] = np.array(pctyu2)/(dx*dt)
w6[0, 2, :, :, 0] = np.array(pctyu3)/(dx*dt)
w7[0, 0, :, :, 0] = np.array(pctyv1)/(dx*dt)
w7[0, 1, :, :, 0] = np.array(pctyv2)/(dx*dt)
w7[0, 2, :, :, 0] = np.array(pctyv3)/(dx*dt)
w8[0, 0, :, :, 0] = np.array(pctyw1)/(dx*dt)
w8[0, 1, :, :, 0] = np.array(pctyw2)/(dx*dt)
w8[0, 2, :, :, 0] = np.array(pctyw3)/(dx*dt)
w9[0, :, :, :, 0] = 0.125

kernel_initializer_1 = tf.keras.initializers.constant(w1)
kernel_initializer_2 = tf.keras.initializers.constant(w2)
kernel_initializer_3 = tf.keras.initializers.constant(w3)
kernel_initializer_4 = tf.keras.initializers.constant(w4)
kernel_initializer_5 = tf.keras.initializers.constant(w5)
kernel_initializer_6 = tf.keras.initializers.constant(w6)
kernel_initializer_7 = tf.keras.initializers.constant(w7)
kernel_initializer_8 = tf.keras.initializers.constant(w8)
kernel_initializer_9 = tf.keras.initializers.constant(w9)
bias_initializer = tf.keras.initializers.constant(np.zeros((1,)))

# ------------------ Load geometry meshing -----------------
mesh = np.load('mesh_64_sk.npy')
# ----------------------------------------------------------

sigma = np.zeros((1, nx, ny, nz, 1)).astype('float32')

# build the model based on the numpy data file
for i in range(1, nx-1):
    for j in range(1, ny-1):
        for k in range(1, nz-1):
            if mesh[0][i+16][j+16][k][0] == 0:
                sigma[0][k][j][i][0] = 100000

# initilization
he = HaloExchange(structured=True, halo_size=1, tensor_used=True,double_precision=True, corner_exchanged=True)
sub_nx, sub_ny, sub_nz, current_domain = he.initialization(sigma, is_periodic=False, is_reordered=False)
sub_x, sub_y, sub_z = sub_nx+2, sub_ny+2, sub_nz+2

current_domain = he.structured_halo_update_3D(current_domain)
current_domain = current_domain.numpy()

rank = he.rank  # get process rank
neighbors = he.neighbors

LEFT = 0
RIGHT = 1
FRONT = 2
BEHIND = 3
TOP = 4
BOTTOM = 5

# ======================== Central differencing =========================================#
# Libraries for solving momentum equation
CNN3D_central_2nd_dif = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_1,
                           bias_initializer=bias_initializer),
])

CNN3D_central_2nd_xadv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_2,
                           bias_initializer=bias_initializer),
])

CNN3D_central_2nd_yadv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_3,
                           bias_initializer=bias_initializer),
])

CNN3D_central_2nd_zadv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_4,
                           bias_initializer=bias_initializer),
])

CNN3D_Su = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_6,
                           bias_initializer=bias_initializer),
])

CNN3D_Sv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_7,
                           bias_initializer=bias_initializer),
])

CNN3D_Sw = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_8,
                           bias_initializer=bias_initializer),
])


CNN3D_pu = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_2,
                           bias_initializer=bias_initializer),
])

CNN3D_pv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_3,
                           bias_initializer=bias_initializer),
])

CNN3D_pw = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_x, sub_y, sub_z, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=kernel_initializer_4,
                           bias_initializer=bias_initializer),
])

# Libraries for solving the Poisson equation

CNN3D_A_66= keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(66, 66, 66, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_34 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(34, 34, 34, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_18 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(18, 18, 18, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_10 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(10, 10, 10, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_6 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(6, 6, 6, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_4 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(4, 4, 4, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_2 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(2, 2, 2, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_1 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(1, 1, 1, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])


# Libraries for solving multi-grid
CNN3D_res_64 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(64, 64, 64, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])
CNN3D_res_32 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(32, 32, 32, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])
CNN3D_res_16 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(16, 16, 16, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])
CNN3D_res_8 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(8, 8, 8, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])

CNN3D_res_4 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(4, 4, 4, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])

CNN3D_res_2 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(2, 2, 2, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])

# Prolongation
CNN3D_prol_64 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(64, 64, 64, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_32 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(32, 32, 32, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_16 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(16, 16, 16, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_8 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(8, 8, 8, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_4 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(4, 4, 4, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_2 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(2, 2, 2, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_1 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(1, 1, 1, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

def boundary_condition_velocity(values_u, values_v, values_w, sub_nx):
    global neighbors, FRONT, BEHIND, LEFT, RIGHT, TOP, BOTTOM
    
    tempu = tf.Variable(values_u)
    tempv = tf.Variable(values_v)
    tempw = tf.Variable(values_w)
    
    # left bound
    if neighbors[LEFT] == -2:
        tempu[0, 1:-1, 1:-1, 1, 0].assign(tf.Variable(tf.ones((1, sub_nx, sub_nx)))[0, :]*ub)
        tempv[0, 1:-1, 1:-1, 1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
        tempw[0, 1:-1, 1:-1, 1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])

    # right bound
    if neighbors[RIGHT] == -2:
        tempu[0, 1:-1, 1:-1, sub_nx,0].assign(tf.Variable(tf.ones((1, sub_nx, sub_nx)))[0, :]*ub)
        tempv[0, 1:-1, 1:-1, sub_nx,0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
        tempw[0, 1:-1, 1:-1, sub_nx,0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])

    # front bound
    if neighbors[FRONT] == -2:
        tempu[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(values_u)[0, 1:-1, 2, 1:-1, 0])
        tempv[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
        tempw[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])

    # back bound
    if neighbors[BEHIND] == -2:
        tempu[0, 1:-1, sub_nx, 1:-1, 0].assign(tf.Variable(values_u)[0, 1:-1, sub_nx-1, 1:-1, 0])
        tempv[0, 1:-1, sub_nx, 1:-1,0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
        tempw[0, 1:-1, sub_nx, 1:-1 ,0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])

    # bottom bound
    if neighbors[BOTTOM] == -2:
        tempu[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
        tempv[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
        tempw[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])

    # top bound
    if neighbors[TOP] == -2:
        tempu[0, sub_nx, 1:-1, 1:-1, 0].assign(tf.Variable(values_u)[0, sub_nx-1, 1:-1, 1:-1, 0])
        tempv[0, sub_nx, 1:-1, 1:-1, 0].assign(tf.Variable(values_v)[0, sub_nx-1, 1:-1, 1:-1, 0])
        tempw[0, sub_nx, 1:-1, 1:-1,0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])

    return tempu, tempv, tempw

def boundary_condition_pressure(values_p, sub_nx):
    global neighbors, FRONT, BEHIND, LEFT, RIGHT, TOP, BOTTOM
    
    tempp = tf.Variable(values_p)
    
    # left right
    if neighbors[RIGHT] == -2:
        tempp[0, 1:-1, 1:-1, sub_nx,0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
    if neighbors[LEFT] == -2:
        tempp[0, 1:-1, 1:-1, 1, 0].assign(tf.Variable(values_p)[0, 1:-1, 1:-1, 2, 0])

    # front behind
    if neighbors[FRONT] == -2:
        tempp[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(values_p)[0, 1:-1, 2, 1:-1, 0])
    if neighbors[BEHIND] == -2:
        tempp[0, 1:-1, sub_nx, 1:-1, 0].assign(tf.Variable(values_p)[0, 1:-1, sub_nx - 1, 1:-1, 0])

    # bottom top
    if neighbors[BOTTOM] == -2:
        tempp[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(values_p)[0, 2, 1:-1, 1:-1, 0])
    if neighbors[TOP] == -2:
        tempp[0, sub_nx, 1:-1, 1:-1, 0].assign(tf.Variable(values_p)[0, sub_nx-1, 1:-1, 1:-1, 0])

    return tempp

def boundary_condition_source(b):
    global neighbors,LEFT
    
    tempb = tf.Variable(b)
    if neighbors[LEFT] == -2:
        tempb[0, :, :, 0, 0].assign(tf.Variable(b)[0, :, :, 1, 0])
    return tempb

def bluff_body(values_u, values_v, values_w, sigma):
    temp1 = values_u / (1+dt*sigma)
    temp2 = values_v / (1+dt*sigma)
    temp3 = values_w / (1+dt*sigma)
    return temp1, temp2, temp3

##############################################################################################################################################

multi_grid_counter = 0

# Initialisation of the CFD model
input_shape = (1, sub_x, sub_y, sub_z, 1)  # (1,66,66,66,1)
values_u = tf.zeros(input_shape)
values_v = tf.zeros(input_shape)
values_w = tf.zeros(input_shape)
values_p = tf.zeros(input_shape)

# ------------------ Numerical set up ----------------------
multi_itr = 10        # Iterations of multi-grid
j_itr = 1             # Iterations of Jacobi
ntime = 100           # Time steps -> should be 100
n_out = 1           # Results output
nrestart = 0          # Last time step for restart
ctime_old = 0         # Last computational time for restart
mgsolver = True       # Multigrid
save_fig = False      # Saving results
Restart = False       # Restart
ctime = 0             # Initialise computational time
# ----------------------------------------------------------

# not sure whether this restart method works on parallel ...
# --------------- Reading previous results -----------------
if Restart == True:
    temp1 = np.load('result_buildings/result_SK_city_Re200/u8000.npy').astype('float32')
    temp2 = np.load('result_buildings/result_SK_city_Re200/v8000.npy').astype('float32')
    temp3 = np.load('result_buildings/result_SK_city_Re200/w8000.npy').astype('float32')
    temp4 = np.load('result_buildings/result_SK_city_Re200/p8000.npy').astype('float32')
    values_u = tf.Variable(values_u)[0, :, :, :, 0].assign(tf.convert_to_tensor(temp1))
    values_v = tf.Variable(values_v)[0, :, :, :, 0].assign(tf.convert_to_tensor(temp2))
    values_w = tf.Variable(values_w)[0, :, :, :, 0].assign(tf.convert_to_tensor(temp3))
    values_p = tf.Variable(values_p)[0, :, :, :, 0].assign(tf.convert_to_tensor(temp4))
    nrestart = 8000
    ctime_old = nrestart*dt
# ----------------------------------------------------------

# solve the problem with Multigrid method
start = perf_counter()

# DOMINANT ITERATIONS
for itime in range(ntime):
    timestep_start = perf_counter()
    ctime = ctime + dt + ctime_old

# ------------------ Boundary conditions ----------------------
    [values_u, values_v, values_w] = boundary_condition_velocity(values_u, values_v, values_w, sub_nx)
    values_p = boundary_condition_pressure(values_p, sub_nx)
    
    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
    values_p = he.structured_halo_update_3D(values_p)
    
# -------------------------------------------------------------

# ------------------ Momentum equations -----------------------

    tempU = tf.reshape(values_u[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempV = tf.reshape(values_v[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempW = tf.reshape(values_w[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))

    a_u = CNN3D_central_2nd_dif(values_u) - \
        tempU*CNN3D_central_2nd_xadv(values_u) - \
        tempV*CNN3D_central_2nd_yadv(values_u) - \
        tempW*CNN3D_central_2nd_zadv(values_u)
    b_u = 0.5*a_u + tempU
    
    
    a_v = CNN3D_central_2nd_dif(values_v) - \
        tempU*CNN3D_central_2nd_xadv(values_v) - \
        tempV*CNN3D_central_2nd_yadv(values_v) - \
        tempW*CNN3D_central_2nd_zadv(values_v)
    b_v = 0.5*a_v + tempV
    
    a_w = CNN3D_central_2nd_dif(values_w) - \
        tempU*CNN3D_central_2nd_xadv(values_w) - \
        tempV*CNN3D_central_2nd_yadv(values_w) - \
        tempW*CNN3D_central_2nd_zadv(values_w)
    b_w = 0.5*a_w + tempW
    
    b_u = he.padding_block_halo_3D(b_u, 1).reshape(1,sub_x,sub_y,sub_z,1)
    b_v = he.padding_block_halo_3D(b_v, 1).reshape(1,sub_x,sub_y,sub_z,1)
    b_w = he.padding_block_halo_3D(b_w, 1).reshape(1,sub_x,sub_y,sub_z,1)
    
    [b_u, b_v, b_w] = boundary_condition_velocity(b_u, b_v, b_w, sub_nx)  # compute boundary velocity
    
    # halo update
    b_u = he.structured_halo_update_3D(b_u)
    b_v = he.structured_halo_update_3D(b_v)
    b_w = he.structured_halo_update_3D(b_w)
    
    tempBU = tf.reshape(b_u[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempBV = tf.reshape(b_v[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempBW = tf.reshape(b_w[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    
    c_u = CNN3D_central_2nd_dif(b_u) - \
        tempBU*CNN3D_central_2nd_xadv(b_u) - \
        tempBV*CNN3D_central_2nd_yadv(b_u) - \
        tempBW*CNN3D_central_2nd_zadv(b_u)

    c_v = CNN3D_central_2nd_dif(b_v) - \
        tempBU*CNN3D_central_2nd_xadv(b_v) - \
        tempBV*CNN3D_central_2nd_yadv(b_v) - \
        tempBW*CNN3D_central_2nd_zadv(b_v)

    c_w = CNN3D_central_2nd_dif(b_w) - \
        tempBU*CNN3D_central_2nd_xadv(b_w) - \
        tempBV*CNN3D_central_2nd_yadv(b_w) - \
        tempBW*CNN3D_central_2nd_zadv(b_w)
        
    tempU = tempU + c_u
    tempV = tempV + c_v
    tempW = tempW + c_w
    
    values_u = he.padding_block_halo_3D(tempU, 1).reshape(1,sub_x,sub_y,sub_z,1)
    values_v = he.padding_block_halo_3D(tempV, 1).reshape(1,sub_x,sub_y,sub_z,1)
    values_w = he.padding_block_halo_3D(tempW, 1).reshape(1,sub_x,sub_y,sub_z,1)
    
    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
    
# -------------------------------------------------------------

# ------------------ Immersed Boundary method -----------------
    [values_u, values_v, values_w] = bluff_body(values_u, values_v, values_w, current_domain)

    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
# -------------------------------------------------------------

# ------------------ Pressure gradient ------------------------

    tempU = tf.reshape(values_u[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempV = tf.reshape(values_v[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempW = tf.reshape(values_w[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    
    tempU = tempU - CNN3D_pu(values_p)
    tempV = tempV - CNN3D_pv(values_p)
    tempW = tempW - CNN3D_pw(values_p)
    
    values_u = he.padding_block_halo_3D(tempU, 1).reshape(1,sub_x,sub_y,sub_z,1)
    values_v = he.padding_block_halo_3D(tempV, 1).reshape(1,sub_x,sub_y,sub_z,1)
    values_w = he.padding_block_halo_3D(tempW, 1).reshape(1,sub_x,sub_y,sub_z,1)
    
    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
    
    [values_u, values_v, values_w] = boundary_condition_velocity(values_u, values_v, values_w, sub_nx)
    
    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)

# -------------------------------------------------------------


# ------------------ Possion equation -------------------------
    b = -(CNN3D_Su(values_u) + CNN3D_Sv(values_v) + CNN3D_Sw(values_w))
    b = boundary_condition_source(b)
# ----------------------------------------------------------------------------------------------------------------------------------

    multi_grid_start = perf_counter()  # multi-grid time counter
    if mgsolver == True:
        for multi_grid in range(multi_itr):
            w = np.zeros([1, 1, 1, 1, 1])
            
            r = CNN3D_A_66(values_p) - b  # compute the residual
            if neighbors[RIGHT] == -2:
                r = tf.Variable(r)[0, :, :, sub_nx-1, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
            
            # restriction
            # r_64 = CNN3D_res_128(r)
            r_32 = CNN3D_res_64(r)
            r_16 = CNN3D_res_32(r_32)
            r_8 = CNN3D_res_16(r_16)
            r_4 = CNN3D_res_8(r_8)
            r_2 = CNN3D_res_4(r_4)
            r_1 = CNN3D_res_2(r_2)

            # prolongation
            for Jacobi in range(j_itr):
                w = (w - CNN3D_A_1(w)/w5[0, 1, 1, 1, 0] + r_1/w5[0, 1, 1, 1, 0])
            w = (w - CNN3D_A_1(w)/w5[0, 1, 1, 1, 0] + r_1/w5[0, 1, 1, 1, 0])
            

            w_2 = CNN3D_prol_1(w)
            w_t1 = he.padding_block_halo_3D(w_2, 1)
            w_t1 = he.structured_halo_update_3D(w_t1)
            for Jacobi in range(j_itr):
                temp2 = CNN3D_A_4(w_t1)
                w_2 = (w_2 - temp2/w5[0, 1, 1, 1, 0] + r_2/w5[0, 1, 1, 1, 0])
                

            w_4 = CNN3D_prol_2(w_2)
            w_t2 = he.padding_block_halo_3D(w_4, 1)
            w_t2 = he.structured_halo_update_3D(w_t2)
            for Jacobi in range(j_itr):
                temp4 = CNN3D_A_6(w_t2)
                w_4 = (w_4 - temp4/w5[0, 1, 1, 1, 0] + r_4/w5[0, 1, 1, 1, 0])

            w_8 = CNN3D_prol_4(w_4)
            w_t3 = he.padding_block_halo_3D(w_8, 1)
            w_t3 = he.structured_halo_update_3D(w_t3)
            for Jacobi in range(j_itr):
                temp8 = CNN3D_A_10(w_t3)
                w_8 = (w_8 - temp8/w5[0, 1, 1, 1, 0] + r_8/w5[0, 1, 1, 1, 0])
                
            w_16 = CNN3D_prol_8(w_8)
            w_t4 = he.padding_block_halo_3D(w_16, 1)
            w_t4 = he.structured_halo_update_3D(w_t4)
            for Jacobi in range(j_itr):
                temp16 = CNN3D_A_18(w_t4)
                w_16 = (w_16 - temp16/w5[0, 1, 1, 1, 0] + r_16/w5[0, 1, 1, 1, 0])
                
            w_32 = CNN3D_prol_16(w_16)
            w_t5 = he.padding_block_halo_3D(w_32, 1)
            w_t5 = he.structured_halo_update_3D(w_t5)
            for Jacobi in range(j_itr):
                temp32 = CNN3D_A_34(w_t5)
                w_32 = (w_32 - temp32/w5[0, 1, 1, 1, 0] + r_32/w5[0, 1, 1, 1, 0])
                
            w_64 = CNN3D_prol_32(w_32)
            w_t6 = he.padding_block_halo_3D(w_64,1)
            w_t6 = he.structured_halo_update_3D(w_t6)  
            for Jacobi in range(j_itr):
                temp64 = CNN3D_A_66(w_t6)
                w_64 = (w_64 - temp64/w5[0, 1, 1, 1, 0]+ r/w5[0, 1, 1, 1, 0])

            w_64 = he.padding_block_halo_3D(w_64,1)
            w_64 = he.structured_halo_update_3D(w_64)

            values_p = values_p - w_64
            if neighbors[RIGHT] == -2:
                values_p = tf.Variable(values_p)[0, 1:-1, 1:-1, sub_nx, 0].assign(tf.Variable(tf.zeros((1, sub_nx, sub_nx)))[0, :])
                
            values_p = he.structured_halo_update_3D(values_p)
            
            tempVal = tf.reshape(values_p[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_nz,1))
            tempVal = tempVal - CNN3D_A_66(values_p)/w5[0, 1, 1, 1, 0] + b/w5[0, 1, 1, 1, 0]
            values_p = he.padding_block_halo_3D(tempVal,1)
            values_p = he.structured_halo_update_3D(values_p)
            

    # count the multigrid runing time
    multi_grid_end = perf_counter()
    multi_grid_counter += (multi_grid_end - multi_grid_start)
    
# -------------------------------------------------------------

# ------------------ Pressure gradient ------------------------
    values_p = boundary_condition_pressure(values_p, sub_nx)
    values_p = he.structured_halo_update_3D(values_p)   # halo update
    
    
    tempU = tf.reshape(values_u[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempV = tf.reshape(values_v[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    tempW = tf.reshape(values_w[0,1:-1,1:-1,1:-1,0],(1,sub_nx,sub_ny,sub_ny,1))
    
    tempU = tempU - CNN3D_pu(values_p) # pressure along x direct
    tempV = tempV - CNN3D_pv(values_p) # pressure along y direct
    tempW = tempW - CNN3D_pw(values_p) # pressure along z direct
    
    values_u = he.padding_block_halo_3D(tempU, 1).reshape(1,sub_x,sub_y,sub_z,1)
    values_v = he.padding_block_halo_3D(tempV, 1).reshape(1,sub_x,sub_y,sub_z,1)
    values_w = he.padding_block_halo_3D(tempW, 1).reshape(1,sub_x,sub_y,sub_z,1)
    
    values_u = he.structured_halo_update_3D(values_u) # halo update
    values_v = he.structured_halo_update_3D(values_v) # halo update
    values_w = he.structured_halo_update_3D(values_w) # halo update
    values_p = he.structured_halo_update_3D(values_p) # halo update
    
# -------------------------------------------------------------

# ------------------ Immersed Boundary method -----------------
    [values_u, values_v, values_w] = boundary_condition_velocity(values_u, values_v, values_w, sub_nx)
    values_u = he.structured_halo_update_3D(values_u) # halo update
    values_v = he.structured_halo_update_3D(values_v) # halo update
    values_w = he.structured_halo_update_3D(values_w) # halo update
    
    [values_u, values_v, values_w] = bluff_body(values_u, values_v, values_w, current_domain)

    values_u = he.structured_halo_update_3D(values_u) # halo update
    values_v = he.structured_halo_update_3D(values_v) # halo update
    values_w = he.structured_halo_update_3D(values_w) # halo update

# -------------------------------------------------------------
    if itime == ntime-1:
        print('ctime', ctime)
    if save_fig == True:
        save_data(values_u, values_v, values_w,values_p, n_out, itime+nrestart)

    timestep_end = perf_counter()
    if he.rank == 0:
        print("[TIME STEP {}] ".format(itime), timestep_end - timestep_start)

# END OF ONE TIMESTEP
end = perf_counter()
print('Total timestepping runtime', (end-start))
total_time = end - start
print('Total multigrid runtime', multi_grid_counter)
print('Runtime mulitigrid/total(%): ', (multi_grid_counter/total_time)*100)

# save the results
np.save('parallel_out/parallel_values_p_{}.npy'.format(rank),values_p[0, 1:-1, 1:-1, 1:-1, 0])
np.save('parallel_out/parallel_values_u_{}.npy'.format(rank),values_u[0, 1:-1, 1:-1, 1:-1, 0])
np.save('parallel_out/parallel_values_v_{}.npy'.format(rank),values_v[0, 1:-1, 1:-1, 1:-1, 0])
np.save('parallel_out/parallel_values_w_{}.npy'.format(rank),values_w[0, 1:-1, 1:-1, 1:-1, 0])
