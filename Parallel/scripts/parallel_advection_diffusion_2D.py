from mpi4py import MPI
# import halo_exchange
from halos_exchange import HaloExchange
import os
import numpy as np  # cupy can be used as optimisation if CUDA/AMD GPUs are available
from tensorflow import keras
import tensorflow as tf
import math
import sys
assert sys.version_info >= (3, 5)
assert tf.__version__ >= "2.0"
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid dnn issue

################################ Load data and initial conditions #######################
# Load data and set initial condition
nx, ny = 300, 300
T = np.zeros([nx, ny], dtype=np.float64)
gamma = 40
# initialise t:
x0 = 0
y0 = -50
x = np.zeros([1, nx],dtype=np.float64)
y = np.zeros([1, ny],dtype=np.float64)

for ii in range(nx):
    x[0][ii] = -150 + 300/nx*ii
    y[0][ii] = -150 + 300/nx*ii

# boundary excluded: range 1-299 x 1-299, I suppose we are using Dirichlet boundary condition
for i in range(1, 299):
    for j in range(1, 299):
        temp1 = -((x[0][i] - x0)**2 + (y[0][j] - y0)**2)
        temp2 = 2*gamma**2
        T[i][j] = math.exp(temp1/temp2)

input_shape = (1, nx, ny, 1)  # (1,300,300,1) as original problem size
values = np.zeros(input_shape,dtype=np.float64) # initalization of the mesh

# generate Gaussian distribution with a blob
for i in range(nx):
    for j in range(ny):
      values[0][i][j][0] = T[i][j]  # + Z1[i][j] + Z2[i][j] + Z3[i][j]*0.5

# generate Gaussian with a blob
for i in range(50):
    for j in range(50):
        values[0][i+225][j+125][0] = values[0][i+225][j+125][0] + 1

################################ MPI Implementation ####################################

# new strategy here
he = HaloExchange(structured=True,tensor_used=True,double_precision=True,corner_exchanged=True)
sub_nx,sub_ny,current_domain = he.initialization(values,is_periodic=False,is_reordered=False)
rank = he.rank
num_process = he.num_process

# 5 stencils
# schemes for advection-term
w1 = ([[[[0.0],        # upwind
        [0.2],
        [0.0]],

        [[0.3],
        [-1.0],
        [0.2]],

        [[0.0],
        [0.3],
        [0.0]]]])

w2 = ([[[[0.0],        # central
        [0.15],
        [0.0]],

        [[0.25],
        [-0.8],
        [0.15]],

        [[0.0],
        [0.25],
        [0.0]]]])


init_kernel_1 = w1
init_kernel_2 = w2

init_bias = np.zeros((1,))  # filters - need change to exact value for bias

kernel_initializer_1 = tf.keras.initializers.constant(
    init_kernel_1)  # initializer which initialize constant tensor
kernel_initializer_2 = tf.keras.initializers.constant(init_kernel_2)

bias_initializer = tf.keras.initializers.constant(init_bias)

# CNN 2D layers: now I generate CNN filters for each subdomains
# filter 1
CNN2D_1 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_nx+2, sub_ny+2, 1)),
    tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',
                          kernel_initializer=kernel_initializer_1,
                          bias_initializer=bias_initializer),
])

# filter 2
CNN2D_2 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sub_nx+2, sub_ny+2, 1)),
    tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',
                          kernel_initializer=kernel_initializer_2,
                          bias_initializer=bias_initializer),
])

# here set up the hyperparameters to tune in the later training process
CNN2D_1.compile(loss="mse",
                optimizer=keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999))
CNN2D_2.compile(loss="mse",
                optimizer=keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999))

# place_holder for final result
result = np.empty((1,))

# do the first update to get the halo values
current_domain = he.structured_halo_update_2D(current_domain)

start_time = MPI.Wtime() # start timer
for t in range(1000):
    # print('CURRENT_DOMAIN_SHAPE: ',current_domain.shape)
    
    # one-step method
    # a = CNN2D_2.predict(current_domain)
    # a = HaloExchange.padding_block_halo_2D(a,1,0)
    # a = tf.convert_to_tensor(a.reshape(1,sub_nx+2,sub_ny+2,1))
    # current_domain += np.copy(a)
    # current_domain = he.structured_halo_update_2D(current_domain)

    # predictor-corrector scheme
    a = CNN2D_2.predict(current_domain)
    a = HaloExchange.padding_block_halo_2D(a,1,0)
    a = a.reshape(1,sub_nx+2,sub_ny+2,1)
    b = (a + current_domain)
    c = (b + current_domain)*0.5
    c = he.structured_halo_update_2D(c) # do one halo update
    d = CNN2D_2.predict(c)
    d = HaloExchange.padding_block_halo_2D(d,1,0)
    d = d.reshape(1,sub_nx+2,sub_ny+2,1)
    current_domain += np.copy(d)
    current_domain = he.structured_halo_update_2D(current_domain)
                                 
    np.save('parallel_steps/AD_2D_proc_{}_parallel_2_step_{}'.format(rank,t),current_domain)

end_time = MPI.Wtime()

# save the sub-domain to corresponding file and then merge them together
result = current_domain.numpy().reshape(sub_nx+2, sub_ny+2) # assign the values to the placeholder
np.save("parallel_out/AD_2D_proc_{}_parallel_{}".format(rank,num_process), result[1:-1,1:-1])

####################################### Terminate the MPI communication ###################################
MPI.Finalize() # optional
print("[INFO] The problem was solved in {} ".format(end_time - start_time))
