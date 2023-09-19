import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Numerical parameters
dx = 0.01
dy = 0.01
dz = 0.01
nx = 128
ny = 128
nz = 128
nlevel = int(math.log(nz,2)) + 1
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
# Define the rest of the filters in a similar way
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
w2[0,0,0,:,:] = p_div_x1/dx
w2[0,0,1,:,:] = p_div_x2/dx
w2[0,0,2,:,:] = p_div_x3/dx
w3[0,0,0,:,:] = p_div_y1/dx
w3[0,0,1,:,:] = p_div_y2/dx
w3[0,0,2,:,:] = p_div_y3/dx
w4[0,0,0,:,:] = p_div_z1/dx 
w4[0,0,1,:,:] = p_div_z2/dx
w4[0,0,2,:,:] = p_div_z3/dx
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
AD2_x = torch.zeros([1,1,3,3,3])
AD2_y = torch.zeros([1,1,3,3,3])
AD2_z = torch.zeros([1,1,3,3,3])
AD2_x[0,0,0,:,:] = curvature_x1/dx**2
AD2_x[0,0,1,:,:] = curvature_x2/dx**2
AD2_x[0,0,2,:,:] = curvature_x3/dx**2
AD2_y[0,0,0,:,:] = curvature_y1/dx**2
AD2_y[0,0,1,:,:] = curvature_y2/dx**2
AD2_y[0,0,2,:,:] = curvature_y3/dx**2
AD2_z[0,0,0,:,:] = curvature_z1/dx**2
AD2_z[0,0,1,:,:] = curvature_z2/dx**2
AD2_z[0,0,2,:,:] = curvature_z3/dx**2

# Restriction filters
w_res = torch.zeros([1,1,2,2,2])
w_res[0,0,:,:,:] = 0.125

# Detecting filters
wxu = torch.zeros([1,1,3,3,3])
wxd = torch.zeros([1,1,3,3,3])
wyu = torch.zeros([1,1,3,3,3])
wyd = torch.zeros([1,1,3,3,3])
wzu = torch.zeros([1,1,3,3,3])
wzd = torch.zeros([1,1,3,3,3])
wxu[0,0,1,1,1] = 1.0
wxu[0,0,1,1,0] = -1.0
wxd[0,0,1,1,1] = -1.0
wxd[0,0,1,1,2] = 1.0
wyu[0,0,1,1,1] = 1.0
wyu[0,0,1,0,1] = -1.0
wyd[0,0,1,1,1] = -1.0
wyd[0,0,1,2,1] = 1.0
wzu[0,0,1,1,1] = 1.0
wzu[0,0,0,1,1] = -1.0
wzd[0,0,1,1,1] = -1.0
wzd[0,0,2,1,1] = 1.0

# Define the PyTorch models for the filters (similar to TensorFlow)
class DMConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 input_shape=None, kernel_initializer=None, bias_initializer=None):
        super(DMConv3D, self).__init__()

        if kernel_initializer is not None:
            depth, height, width = input_shape[1], input_shape[2], input_shape[3]

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

        # Initialize the weights with your custom initializer
        # for trainable weights, use nn.Parameters instead.
        if kernel_initializer is not None:
            self.conv3d.weight.data = kernel_initializer

        # Initialize the biases with your custom initializer
        if bias_initializer is not None:
            self.conv3d.bias.data = bias_initializer
       
        # Calculate the output shape based on the input shape and convolution parameters
        #with torch.no_grad():
        #    input_tensor = torch.randn(1, in_channels, *input_shape)
        #    output_tensor = self.conv3d(input_tensor)
        # 
        #self.output_shape = tuple(output_tensor.shape[1:])  # Exclude batch size

    def forward(self, x):
        return self.conv3d(x)

# Usage of Conv3D layers with initialized kernels
difx = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = AD2_x,
                bias_initializer = bias_initializer)
dify = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = AD2_y,
                bias_initializer = bias_initializer)
difz = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = AD2_z,
                bias_initializer = bias_initializer)

dif = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = w1,
                bias_initializer = bias_initializer)

xadv = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = w2,
                bias_initializer = bias_initializer)
yadv = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = w3,
                bias_initializer = bias_initializer)
zadv = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = w4,
                bias_initializer = bias_initializer)

wxu= DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = wxu,
                bias_initializer = bias_initializer)
wxd= DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = wxd,
                bias_initializer = bias_initializer) 
wyu= DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = wyu,
                bias_initializer = bias_initializer)
wyd= DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = wyd,
                bias_initializer = bias_initializer)
wzu= DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = wzu,
                bias_initializer = bias_initializer)
wzd= DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = wzd,
                bias_initializer = bias_initializer)

# Similarly, define other layers and models for your filters
for i in range(nlevel-1):
    locals()['res_'+str(2**(i+1))] = DMConv3D(1, 1, kernel_size=2, stride=2, padding=0,
                input_shape=(1, int(nz*0.5**(nlevel-2-i)), int(ny*0.5**(nlevel-2-i)), int(nx*0.5**(nlevel-2-i))),
                kernel_initializer = w_res,
                bias_initializer = bias_initializer)

print("nlevel in filter =", nlevel)
for i in range(nlevel):
       print("in the loop, nlevel= ", nlevel, 'A_'+str(2**i)) 
       if i < nlevel-1:
              locals()['A_'+str(2**i)] = DMConv3D(1, 1, kernel_size=3, stride=1, padding='same',
                input_shape=(1, int(nz*0.5**(nlevel-1-i)), int(ny*0.5**(nlevel-1-i)), int(nx*0.5**(nlevel-1-i))),
                kernel_initializer = wA,
                bias_initializer = bias_initializer)
       else:
              locals()['A_'+str(2**i)] = DMConv3D(1, 1, kernel_size=3, stride=1, padding=0,
                input_shape=(1, nz+2, ny+2, nx+2),
                kernel_initializer = wA,
                bias_initializer = bias_initializer)

for i in range(nlevel-1):
    input_shape=(1, 1*2**i, 1*1*2**i, 1*1*2**i)
    locals()['prol_'+str(2**i)] = nn.Sequential(
         nn.Upsample(scale_factor=2, mode='nearest'),
    )