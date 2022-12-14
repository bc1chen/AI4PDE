# TensorFlow ≥2.0 is required
from time import perf_counter
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # avoid tf dnn flag issue


# flags and env variables
# if we want compile and run as tf.function
tf.config.run_functions_eagerly(True)

# https://cupy.dev/ # accelarate the mult-dim array on GPU
# def cupy2tensor(cp_array):
#   cap = cp_array.toDlpack()
#   return tf.experimental.dlpack.from_dlpack(cap)

# def tensor2cupy(tensor):
#   cap = tf.experimental.dlpack.to_dlpack(tensor)
#   return cp.from_dlpack(cap)

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

# Libraries for solving momentum equation
CNN3D_central_2nd_dif = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_1,
                           bias_initializer=bias_initializer),
])

CNN3D_central_2nd_xadv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_2,
                           bias_initializer=bias_initializer),
])

CNN3D_central_2nd_yadv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_3,
                           bias_initializer=bias_initializer),
])

CNN3D_central_2nd_zadv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_4,
                           bias_initializer=bias_initializer),
])

CNN3D_Su = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_6,
                           bias_initializer=bias_initializer),
])

CNN3D_Sv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_7,
                           bias_initializer=bias_initializer),
])

CNN3D_Sw = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_8,
                           bias_initializer=bias_initializer),
])


CNN3D_pu = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_2,
                           bias_initializer=bias_initializer),
])

CNN3D_pv = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_3,
                           bias_initializer=bias_initializer),
])

CNN3D_pw = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',
                           kernel_initializer=kernel_initializer_4,
                           bias_initializer=bias_initializer),
])

# Libraries for solving the Poisson equation
CNN3D_A_512 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_256 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(256, 256, 256, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_128 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(128, 128, 128, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_64 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(64, 64, 64, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_32 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(32, 32, 32, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_16 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(16, 16, 16, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_8 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(8, 8, 8, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                           kernel_initializer=kernel_initializer_5,
                           bias_initializer=bias_initializer),
])

CNN3D_A_4 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(4, 4, 4, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
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
CNN3D_res_512 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, nz, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])

CNN3D_res_256 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(256, 256, 256, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])

CNN3D_res_128 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(128, 128, 128, 1)),
    tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                           kernel_initializer=kernel_initializer_9,
                           bias_initializer=bias_initializer),
])

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

CNN3D_prol_256 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(256, 256, 256, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

CNN3D_prol_128 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(128, 128, 128, 1)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])

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

# Functions linking to the AI libraries

def boundary_condition_velocity(values_u, values_v, values_w, nx):
    tempu = tf.Variable(values_u)
    tempv = tf.Variable(values_v)
    tempw = tf.Variable(values_w)

    tempu[0, :, :, 0, 0].assign(tf.Variable(tf.ones((1, nx, nx)))[0, :]*ub)
    tempv[0, :, :, 0, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    tempw[0, :, :, 0, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])

    tempu[0, :, :, nx-1, 0].assign(tf.Variable(tf.ones((1, nx, nx)))[0, :]*ub)
    tempv[0, :, :, nx-1, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    tempw[0, :, :, nx-1, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])

    tempu[0, :, 0, :, 0].assign(tf.Variable(values_u)[0, :, 1, :, 0])
    tempv[0, :, 0, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    tempw[0, :, 0, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])

    tempu[0, :, nx-1, :, 0].assign(tf.Variable(values_u)[0, :, nx-2, :, 0])
    tempv[0, :, nx-1, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    tempw[0, :, nx-1, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])

    tempu[0, 0, :, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    tempv[0, 0, :, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    tempw[0, 0, :, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])

    tempu[0, nx-1, :, :, 0].assign(tf.Variable(values_u)[0, nx-2, :, :, 0])
    tempv[0, nx-1, :, :, 0].assign(tf.Variable(values_v)[0, nx-2, :, :, 0])
    tempw[0, nx-1, :, :, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    return tempu, tempv, tempw

def boundary_condition_pressure(values_p, nx):
    tempp = tf.Variable(values_p)
    tempp[0, :, :, nx-1, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
    tempp[0, :, :, 0, 0].assign(tf.Variable(values_p)[0, :, :, 1, 0])

    tempp[0, :, 0, :, 0].assign(tf.Variable(values_p)[0, :, 1, :, 0])
    tempp[0, :, nx-1, :, 0].assign(tf.Variable(values_p)[0, :, nx-2, :, 0])

    tempp[0, 0, :, :, 0].assign(tf.Variable(values_p)[0, 1, :, :, 0])
    tempp[0, nx-1, :, :, 0].assign(tf.Variable(values_p)[0, nx-2, :, :, 0])
    return tempp

def boundary_condition_source(b):
    tempb = tf.Variable(b)
    tempb[0, :, :, 0, 0].assign(tf.Variable(b)[0, :, :, 1, 0])

    # return tf.convert_to_tensor(b.reshape(1,nx, ny, nz, 1)) # return as tensor
    return tempb

def bluff_body(values_u, values_v, values_w, sigma):
    temp1 = values_u / (1+dt*sigma)
    temp2 = values_v / (1+dt*sigma)
    temp3 = values_w / (1+dt*sigma)

    return temp1, temp2, temp3

def save_data(values_u, values_v, values_w, values_p, n_out, itime):
    if itime % n_out == 0:
        np.save("result_buildings/result_SK_city_Re200/u" + str(itime), arr=values_u[0, :, :, :, 0])
        np.save("result_buildings/result_SK_city_Re200/v" + str(itime), arr=values_v[0, :, :, :, 0])
        np.save("result_buildings/result_SK_city_Re200/w" + str(itime), arr=values_w[0, :, :, :, 0])
        np.save("result_buildings/result_SK_city_Re200/p" + str(itime), arr=values_p[0, :, :, :, 0])

##############################################################################################################################################


multi_grid_counter = 0

# Initialisation of the CFD model
input_shape = (1, nx, ny, nz, 1)
values_u = tf.zeros(input_shape)
values_v = tf.zeros(input_shape)
values_w = tf.zeros(input_shape)
values_p = tf.zeros(input_shape)
sigma = np.zeros(input_shape).astype('float32')

# ------------------ Numerical set up ----------------------
multi_itr = 10        # Iterations of multi-grid
j_itr = 1             # Iterations of Jacobi
ntime = 100           # Time steps -> should be 100
n_out = 500           # Results output
nrestart = 0          # Last time step for restart
ctime_old = 0         # Last computational time for restart
mgsolver = True       # Multigrid
save_fig = False      # Saving results
Restart = False       # Restart
ctime = 0             # Initialise computational time
# ----------------------------------------------------------

# ------------------ Load geometry meshing -----------------
mesh = np.load('mesh_64_sk.npy')
# ----------------------------------------------------------

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

print('--------------- General numerical parameters ----------------')
print('Computational domain resolution:', values_u.shape)
print('Mesh resolution:', mesh.shape)
print('Time step:', ntime)
print('Initial time:', ctime)
print('-------------------------------------------------------------')

# build the model based on the numpy data file
for i in range(1, nx-1):
    for j in range(1, ny-1):
        for k in range(1, nz-1):
            if mesh[0][i+16][j+16][k][0] == 0:
                sigma[0][k][j][i][0] = 100000

sigma = tf.convert_to_tensor(sigma)

# solve the problem with Multigrid method
start = perf_counter()
for itime in range(ntime):
    ctime = ctime + dt + ctime_old
# ------------------ Boundary conditions ----------------------
    [values_u, values_v, values_w] = boundary_condition_velocity(
        values_u, values_v, values_w, nx)
    values_p = boundary_condition_pressure(values_p, nx)
# -------------------------------------------------------------

# ------------------ Momentum equations -----------------------
    a_u = CNN3D_central_2nd_dif(values_u) - \
        values_u*CNN3D_central_2nd_xadv(values_u) - \
        values_v*CNN3D_central_2nd_yadv(values_u) - \
        values_w*CNN3D_central_2nd_zadv(values_u)
    b_u = 0.5*a_u + values_u

    a_v = CNN3D_central_2nd_dif(values_v) - \
        values_u*CNN3D_central_2nd_xadv(values_v) - \
        values_v*CNN3D_central_2nd_yadv(values_v) - \
        values_w*CNN3D_central_2nd_zadv(values_v)
    b_v = 0.5*a_v + values_v

    a_w = CNN3D_central_2nd_dif(values_w) - \
        values_u*CNN3D_central_2nd_xadv(values_w) - \
        values_v*CNN3D_central_2nd_yadv(values_w) - \
        values_w*CNN3D_central_2nd_zadv(values_w)
    b_w = 0.5*a_w + values_w

    [b_u, b_v, b_w] = boundary_condition_velocity(
        b_u, b_v, b_w, nx)  # compute boundary velocity

    c_u = CNN3D_central_2nd_dif(b_u) - \
        b_u*CNN3D_central_2nd_xadv(b_u) - \
        b_v*CNN3D_central_2nd_yadv(b_u) - \
        b_w*CNN3D_central_2nd_zadv(b_u)

    c_v = CNN3D_central_2nd_dif(b_v) - \
        b_u*CNN3D_central_2nd_xadv(b_v) - \
        b_v*CNN3D_central_2nd_yadv(b_v) - \
        b_w*CNN3D_central_2nd_zadv(b_v)

    c_w = CNN3D_central_2nd_dif(b_w) - \
        b_u*CNN3D_central_2nd_xadv(b_w) - \
        b_v*CNN3D_central_2nd_yadv(b_w) - \
        b_w*CNN3D_central_2nd_zadv(b_w)

    values_u = values_u + c_u
    values_v = values_v + c_v
    values_w = values_w + c_w

# -------------------------------------------------------------

# ------------------ Immersed Boundary method -----------------
    [values_u, values_v, values_w] = bluff_body(
        values_u, values_v, values_w, sigma)
# -------------------------------------------------------------

# ------------------ Pressure gradient ------------------------
    values_u = values_u - CNN3D_pu(values_p)
    values_v = values_v - CNN3D_pv(values_p)
    values_w = values_w - CNN3D_pw(values_p)
    [values_u, values_v, values_w] = boundary_condition_velocity(
        values_u, values_v, values_w, nx)
# -------------------------------------------------------------

# ------------------ Possion equation -------------------------

    # possion equation (multi-grid) A*P = Su
    b = -(CNN3D_Su(values_u) + CNN3D_Sv(values_v) + CNN3D_Sw(values_w))
    b = boundary_condition_source(b)

    # multi-grid method
    multi_grid_start = perf_counter()
    if mgsolver == True:
        for multi_grid in range(multi_itr):
            w_2 = tf.zeros([1, 2, 2, 2, 1])
            r = CNN3D_A_128(values_p) - b
            r = tf.Variable(r)[0, :, :, nx-1, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])

            # restriction
            r_64 = CNN3D_res_128(r)
            r_32 = CNN3D_res_64(r_64)
            r_16 = CNN3D_res_32(r_32)
            r_8 = CNN3D_res_16(r_16)
            r_4 = CNN3D_res_8(r_8)
            r_2 = CNN3D_res_4(r_4)
            # r1 = CNN3D_res_2(r_2)

            # prolongation
            # for Jacobi in range(j_itr):
            #     w = (w - CNN3D_A_1(w)/w5[0,1,1,1,0] + r1/w5[0,1,1,1,0])
            # w_2 = CNN3D_prol_1(w)

            for Jacobi in range(j_itr):
                w_2 = (w_2 - CNN3D_A_2(w_2) /w5[0, 1, 1, 1, 0] + r_2/w5[0, 1, 1, 1, 0])

            w_4 = CNN3D_prol_2(w_2)
            for Jacobi in range(j_itr):
                w_4 = (w_4 - CNN3D_A_4(w_4) /w5[0, 1, 1, 1, 0] + r_4/w5[0, 1, 1, 1, 0])

            w_8 = CNN3D_prol_4(w_4)
            for Jacobi in range(j_itr):
                w_8 = (w_8 - CNN3D_A_8(w_8) /w5[0, 1, 1, 1, 0] + r_8/w5[0, 1, 1, 1, 0])

            w_16 = CNN3D_prol_8(w_8)
            for Jacobi in range(j_itr):
                w_16 = (w_16 - CNN3D_A_16(w_16) /w5[0, 1, 1, 1, 0] + r_16/w5[0, 1, 1, 1, 0])

            w_32 = CNN3D_prol_16(w_16)
            for Jacobi in range(j_itr):
                w_32 = (w_32 - CNN3D_A_32(w_32) /w5[0, 1, 1, 1, 0] + r_32/w5[0, 1, 1, 1, 0])

            w_64 = CNN3D_prol_32(w_32)
            for Jacobi in range(j_itr):
                w_64 = (w_64 - CNN3D_A_64(w_64) /w5[0, 1, 1, 1, 0] + r_64/w5[0, 1, 1, 1, 0])

            w_128 = CNN3D_prol_64(w_64)
            w_128 = (w_128 - CNN3D_A_128(w_128) /w5[0, 1, 1, 1, 0] + r/w5[0, 1, 1, 1, 0])

            # correction
            values_p = values_p - w_128
            values_p = tf.Variable(values_p)[0, :, :, nx-1, 0].assign(tf.Variable(tf.zeros((1, nx, nx)))[0, :])
            values_p = (values_p - CNN3D_A_128(values_p) /w5[0, 1, 1, 1, 0] + b/w5[0, 1, 1, 1, 0])

# -------------------------------------------------------------
    multi_grid_end = perf_counter()
    multi_grid_counter += (multi_grid_end - multi_grid_start)

# ------------------ Pressure gradient ------------------------
    values_p = boundary_condition_pressure(values_p, nx)

    np.save("boundary_conditions/pg1_val_p.npy", values_p)

    values_u = values_u - CNN3D_pu(values_p)  # pressure along x direct
    values_v = values_v - CNN3D_pv(values_p)  # pressure along y direct
    values_w = values_w - CNN3D_pw(values_p)  # pressure along z direct

    np.save("boundary_conditions/pg1_val_u.npy", values_u)
    np.save("boundary_conditions/pg1_val_v.npy", values_v)
    np.save("boundary_conditions/pg1_val_w.npy", values_w)
# -------------------------------------------------------------

# ------------------ Immersed Boundary method -----------------
    [values_u, values_v, values_w] = boundary_condition_velocity(
        values_u, values_v, values_w, nx)
    [values_u, values_v, values_w] = bluff_body(
        values_u, values_v, values_w, sigma)
# -------------------------------------------------------------
    if itime == ntime-1:
        print('ctime', ctime)
    if save_fig == True:
        save_data(values_u, values_v, values_w,
                  values_p, n_out, itime+nrestart)

    print("[TIME STEP {}] ".format(itime))

# END OF TIMESTEP HERE
end = perf_counter()
total_time = end - start
print('Total timestepping runtime', total_time)
print('Total multigrid runtime', multi_grid_counter)
print('Runtime mulitigrid/total(%): ', (multi_grid_counter/total_time)*100)

# save the result
np.save('serial_out/serial_values_p.npy', values_p)
np.save('serial_out/serial_values_u.npy', values_u)
np.save('serial_out/serial_values_v.npy', values_v)
np.save('serial_out/serial_values_w.npy', values_w)
