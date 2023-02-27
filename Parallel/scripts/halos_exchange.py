# imports
import sys
import numpy as np  # cupy can be used as optimisation if CUDA/AMD GPUs are available
import tensorflow as tf
from mpi4py import MPI

# default global settings
assert tf.__version__ >= "2.0"
np.set_printoptions(threshold=sys.maxsize)  # print out the full numpy array
# np.set_printoptions(suppress=True)

# Halo Exchange class


class HaloExchange:
    def __init__(self, structured=True, halo_size=1, ndim=1, tensor_used=False, double_precision=False, corner_exchanged=False) -> None:
        self.comm = None
        self.rank = 0
        self.num_process = 1
        self.rows, self.cols = 1, 1
        self.sub_nx, self.sub_ny, self.sub_nz = 0, 0, 0
        self.topology_dim = -1
        self.neighbors = [-2] * 26  # suppose we support up to 3D, 26 neighbors
        self.current_vector = None
        self.current_matrix = None
        self.current_cuboid = None
        self.halo_size = halo_size  # halo size of the decomposed domains default by 1
        self.data_type = MPI.INT  # set default MPI datatype for MPI communicating
        self.structured_mesh = structured
        self.is_tensor_mesh = tensor_used
        self.is_double_precision = double_precision
        self.ndims = ndim
        # do we need to have the diagonally neighboring tiles exchanged
        self.is_corner_exchanged = corner_exchanged
        self.TOP, self.BOTTOM, self.LEFT, self.RIGHT, self.FRONT, self.BEHIND = 0, 0, 0, 0, 0, 0

        if self.is_double_precision:
            self.data_type = MPI.DOUBLE_PRECISION

    def __repr__(self):
        return f"""
        simultas.HaloExchange instance summary info:
        ===========================================
        [Number of processes] {self.num_process}
        [Processor rank] {self.rank}
        [Topology Dimension] {self.topology_dim}
        [Subdomain shape] ({self.sub_nx+2}, {self.sub_ny+2}, {self.sub_nz+2})
        [Connected neighbors] {[idx for idx in self.neighbors if idx != -2]}
        [Mesh dimension] {self.ndims}
        [Decomposed 1D domain] at {hex(id(self.current_vector))}
        [Decomposed 2D domain] at {hex(id(self.current_matrix))}
        [Decomposed 3D domain] at {hex(id(self.current_cuboid))}
        [Halo Layers] {self.halo_size} 
        [Structured Mesh] {self.structured_mesh}
        [Double Precision] {self.is_double_precision}
        [Corner-neighboring tiles included] {self.is_corner_exchanged}
        ===========================================
        """

    ########## member functions ################
    def mpi_init(self, proc_grid_dim, is_periodic, is_reordered) -> None:
        """Initialize the MPI topology and other MPI processes

        Args:
            proc_grid_dim (tuple): 2D or 3D cartesian grid coordinate in format (x,y) or (z,x,y)
            is_periodic (bool): flag to control periodic boundary condition
            is_reordered (bool): flag to control the cartesian topology is reordered

        Returns:
            None
        """
        # create Cartesian Topology
        self.comm = MPI.COMM_WORLD.Create_cart(
            proc_grid_dim,
            periods=is_periodic,  # set True if periodic boundary
            reorder=is_reordered)

        # get the rank of current process
        self.rank = self.comm.rank

        # TODO: Other topologies if we need

    # TODO: make the init func do much work as possible
    def initialization(self, mesh, topology=None, is_periodic=False, is_reordered=False):
        """_summary_

        Args:
            mesh (_type_): _description_
            topology (_type_, optional): _description_. Defaults to None.product of topology equals to the number of processes
            is_periodic (bool, optional): _description_. Defaults to False.
            is_reordered (bool, optional): _description_. Defaults to False.

        Returns:
            NotImplementedError()
        """
        # convert to numpy array if it is tensor
        if tf.is_tensor(mesh):
            mesh = mesh.numpy()

        # remove the extra one dimensions
        mesh = HaloExchange.remove_one_dims(mesh,expected_dim=self.ndims)
        self.num_process = MPI.COMM_WORLD.Get_size()

        if topology == None:
            # default place_holder for 1D case
            self.topology_dim = (self.num_process, 1)

            if mesh.ndim == 2:
                self.ndims = 2
                self.topology_dim = HaloExchange.generate_proc_dim_2D(self.num_process)
            elif mesh.ndim == 3:
                self.ndims = 3
                self.topology_dim = HaloExchange.generate_proc_dim_3D(self.num_process)
        else:
            self.topology_dim = topology

        assert self.num_process > 1, f"[WARNING] Parallelisation involves 2 or more processes, otherwise run code without MPI (in serial)."

        # MPI initialization
        self.mpi_init(self.topology_dim, is_periodic, is_reordered)  # mpi initialization
        
        print(f'[INFO] Processor {self.rank} activated !')
        
        print(self.topology_dim)

        # return the decomposed sub-domain
        if mesh.ndim == 1:
            return self.domain_decomposition_1D(mesh, mesh.shape[0])
        elif mesh.ndim == 2:
            return self.domain_decomposition_2D(mesh, mesh.shape[0], mesh.shape[1], self.topology_dim)
        elif mesh.ndim == 3:
            return self.domain_decomposition_3D(mesh, mesh.shape[0], mesh.shape[1], mesh.shape[2], self.topology_dim)

        # for dimensions > 3, return nothing for now
        return NotImplementedError()

    # clear cached mesh data
    def clear_cache(self) -> None:
        """clear all cached problem domains

        Returns:
            None
        """
        del self.current_vector
        del self.current_matrix
        del self.current_cuboid

    # decompose 1D block structured mesh
    def domain_decomposition_1D(self, mesh, nx):
        """domain decomposition for 1D block structured meshes

        Args:
            mesh (numpy array): the input problem mesh
            nx (int): shape of the input problem mesh (nx,)

        Returns:
            int, numpy array: single sub-domain shape, sub-domains
        """
        self.LEFT, self.RIGHT = 0, 1
        self.neighbors[self.LEFT], self.neighbors[self.RIGHT] = self.comm.Shift(0, 1)

        sub_domains = HaloExchange.domain_decomposition_strip(mesh.reshape(nx,), self.num_process)  # (1,x)
        self.sub_nx = sub_domains[self.rank].shape[0]
        self.current_vector = np.pad(sub_domains[self.rank], (self.halo_size, self.halo_size), "constant", constant_values=(0,))

        return self.sub_nx, self.current_vector

    # decompose 2D block structured mesh
    def domain_decomposition_2D(self, mesh, nx, ny, proc_grid_dim):
        """domain decomposition for 2D block structured meshes

        Args:
            mesh (numpy array): problem mesh
            nx (int): x shape of the problem mesh
            ny (int): y shape of the problem mesh
            proc_grid_dim (np.array)

        Returns:
            int,int,numpy array: sub-domain shape x, sub-domain shape y, sub-domains
        """
        self.TOP, self.BOTTOM, self.LEFT, self.RIGHT = 0, 1, 2, 3

        if proc_grid_dim[0] == 1 or proc_grid_dim[1] == 1:
            sub_domains = HaloExchange.domain_decomposition_strip(mesh.reshape(nx, ny), self.num_process)  # 2 process
            self.sub_nx, self.sub_ny = sub_domains[0].shape
        else:
            # if the process arrays is 2D then use grid decomposition to split domain
            sub_domains = HaloExchange.domain_decomposition_grid(mesh.reshape(nx, ny), proc_grid_dim)
            self.sub_nx, self.sub_ny = sub_domains[0].shape

        # find the processor id of all neighboring processors
        self.neighbors[self.TOP], self.neighbors[self.BOTTOM] = self.comm.Shift(0, 1)
        self.neighbors[self.LEFT], self.neighbors[self.RIGHT] = self.comm.Shift(1, 1)

        self.current_matrix = np.pad(sub_domains[self.rank], (self.halo_size, self.halo_size), "constant", constant_values=(0,))

        return self.sub_nx, self.sub_ny, self.current_matrix

    # decompose 3d block structured data
    def domain_decomposition_3D(self, mesh, nx, ny, nz, proc_grid_dim):
        """domain decomposition for 3D block structured meshes

        Args:
            mesh (numpy array): problem mesh
            nx (int): x shape of the problem mesh
            ny (int): y shape of the problem mesh
            nz (int): z shape of the problem mesh
            proc_grid_dim(np.array): Cartesian topology

        Returns:
            int,int,int,numpy array: sub-domain shape x, sub-domain shape y, sub-domain shape z, sub-domains
        """
        # neighbor indices
        self.LEFT, self.RIGHT, self.FRONT, self.BEHIND, self.TOP, self.BOTTOM = 0, 1, 2, 3, 4, 5

        # edge case, if 1 process we do nothing
        if self.num_process == 1:
            return nx, ny, nz, mesh

        sub_cubes = HaloExchange.domain_decomposition_cube(mesh.reshape(nx, ny, nz), proc_grid_dim)  # if it is numpy reshape directly
        
        # padding the halo grids
        self.current_cuboid = np.pad(sub_cubes[self.rank], (self.halo_size, self.halo_size), 'constant', constant_values=(0,))

        self.sub_nx, self.sub_ny, self.sub_nz = sub_cubes[0].shape
        # find neighbors (note here 0,1,2 are x,y,z coordinates respectively)
        self.neighbors[self.LEFT], self.neighbors[self.RIGHT] = self.comm.Shift(2, 1)
        self.neighbors[self.FRONT], self.neighbors[self.BEHIND] = self.comm.Shift(1, 1)
        self.neighbors[self.BOTTOM], self.neighbors[self.TOP] = self.comm.Shift(0, 1)

        # return tf.convert_to_tensor(current_cube,np.float64)
        return self.sub_nx, self.sub_ny, self.sub_nz, self.current_cuboid

    # halo exchange in structured 1D
    def structured_halo_update_1D(self, input_vector):
        """parallel updating of halos in 1D

        Args:
            input_vector (numpy): 1D sub-domain

        Returns:
            tensor: 1D tensorflow tensor with halos updated
        """

        if tf.is_tensor(input_vector):
            self.current_vector = input_vector.numpy()
        else:
            self.current_vector = input_vector

        if self.current_vector.ndim > 1:
            self.current_vector = np.squeeze(self.current_vector, axis=0)
            self.current_vector = np.squeeze(self.current_vector, axis=-1)

        self.LEFT, self.RIGHT = 0, 1
        self.sub_nx = self.current_vector.shape[0]

        send_left = np.copy(np.ascontiguousarray(self.current_vector[1]))
        send_right = np.copy(np.ascontiguousarray(self.current_vector[-2]))

        [recv_right, recv_left] = self.halo_update_non_blocking(send_buffers=[send_left, send_right], neighbor_indices=[self.LEFT, self.RIGHT])

        if self.neighbors[self.RIGHT] != -2:
            self.current_vector[-self.halo_size:] = recv_right
        if self.neighbors[self.LEFT] != -2:
            self.current_vector[0:self.halo_size] = recv_left

        if self.is_tensor_mesh:
            return tf.convert_to_tensor(self.current_vector.reshape(1, self.sub_nx, 1))

        return self.current_vector

    # halo exchange in structured 2D
    def structured_halo_update_2D(self, input_matrix):
        """parallel updating of halos in 2D

        Args:
            input_domain (numpy): 2D sub-domain

        Returns:
            tensor: 2D tensorflow tensor with halos updated
        """
        # update the values of the domain
        if tf.is_tensor(input_matrix):
            self.current_matrix = np.copy(input_matrix.numpy())
        else:
            self.current_matrix = np.copy(input_matrix)

        # remove trivial one dimensions
        if self.current_matrix.ndim > 2:
            self.current_matrix = HaloExchange.remove_one_dims(self.current_matrix)

        # get sub-domain shape
        self.sub_nx, self.sub_ny = self.current_matrix.shape

        # neighbor indices
        self.TOP, self.BOTTOM, self.LEFT, self.RIGHT = 0, 1, 2, 3

        # left and right
        send_left = np.copy(np.ascontiguousarray(self.current_matrix[self.halo_size:-self.halo_size, self.halo_size:self.halo_size+self.halo_size]))
        send_right = np.copy(np.ascontiguousarray(self.current_matrix[self.halo_size:-self.halo_size, -self.halo_size-self.halo_size:-self.halo_size]))

        [recv_right, recv_left] = self.halo_update_non_blocking(send_buffers=[send_left, send_right], neighbor_indices=[self.LEFT, self.RIGHT])

        if self.neighbors[self.RIGHT] != -2:
            self.current_matrix[self.halo_size:-self.halo_size, -self.halo_size:] = recv_right
        if self.neighbors[self.LEFT] != -2:
            self.current_matrix[self.halo_size:-self.halo_size, 0:self.halo_size] = recv_left

        send_top = np.copy(np.ascontiguousarray(self.current_matrix[self.halo_size:self.halo_size+self.halo_size, :]))
        send_bottom = np.copy(np.ascontiguousarray(self.current_matrix[-self.halo_size-self.halo_size:-self.halo_size, :]))

        [recv_bottom, recv_top] = self.halo_update_non_blocking(send_buffers=[send_top, send_bottom], neighbor_indices=[self.TOP, self.BOTTOM])

        if self.neighbors[self.TOP] != -2:
            self.current_matrix[0:self.halo_size, :] = recv_top
        if self.neighbors[self.BOTTOM] != -2:
            self.current_matrix[-self.halo_size:, :] = recv_bottom

        # if tensor used for conv, add one dimensions on both edges
        if self.is_tensor_mesh:
            return tf.convert_to_tensor(self.current_matrix.reshape(1, self.sub_nx, self.sub_ny, 1))

        # return current_domain
        return self.current_matrix

    # halo exchange in structured 3D
    def structured_halo_update_3D(self, input_cube):
        """parallel updating of halos in 3D

        Args:
            input_cube (numpy): 3D sub-domain to be updated

        Returns:
            tensor: 3D tensorflow tensors with halos updated 
        """
        # update the values of the domain
        if tf.is_tensor(input_cube):
            self.current_cuboid = np.copy(input_cube.numpy())
        else:
            self.current_cuboid = np.copy(input_cube)

        if self.current_cuboid.ndim > 3:
            self.current_cuboid = HaloExchange.remove_one_dims(self.current_cuboid,self.ndims)

        self.sub_nx, self.sub_ny, self.sub_nz = self.current_cuboid.shape

        # neighbor indices
        self.LEFT, self.RIGHT, self.FRONT, self.BEHIND, self.TOP, self.BOTTOM = 0, 1, 2, 3, 4, 5

        sendbuffer_1 = np.copy(np.ascontiguousarray(self.current_cuboid[self.halo_size:-self.halo_size, self.halo_size: 2*self.halo_size, self.halo_size:-self.halo_size]))
        sendbuffer_2 = np.copy(np.ascontiguousarray(self.current_cuboid[self.halo_size:-self.halo_size, -2*self.halo_size:-self.halo_size, self.halo_size:-self.halo_size]))

        [recvbuffer_1, recvbuffer_2] = self.halo_update_non_blocking(send_buffers=[sendbuffer_1, sendbuffer_2], neighbor_indices=[self.FRONT, self.BEHIND])

        # update front and behind
        if self.neighbors[self.FRONT] != -2:
            self.current_cuboid[self.halo_size:-self.halo_size, 0:self.halo_size, self.halo_size:-self.halo_size] = recvbuffer_2
        if self.neighbors[self.BEHIND] != -2:
            self.current_cuboid[self.halo_size:-self.halo_size, -self.halo_size:, self.halo_size:-self.halo_size] = recvbuffer_1

        sendbuffer_1 = np.copy(np.ascontiguousarray(self.current_cuboid[:, :, self.halo_size:2*self.halo_size]))
        sendbuffer_2 = np.copy(np.ascontiguousarray(self.current_cuboid[:, :, -2*self.halo_size:-self.halo_size]))

        [recvbuffer_1, recvbuffer_2] = self.halo_update_non_blocking(send_buffers=[sendbuffer_1, sendbuffer_2], neighbor_indices=[self.LEFT, self.RIGHT])

        if self.neighbors[self.LEFT] != -2:
            self.current_cuboid[:, :, 0:self.halo_size] = recvbuffer_2
        if self.neighbors[self.RIGHT] != -2:
            self.current_cuboid[:, :, -self.halo_size:] = recvbuffer_1

        sendbuffer_1 = np.copy(np.ascontiguousarray(self.current_cuboid[-2*self.halo_size:-self.halo_size, :, :]))
        sendbuffer_2 = np.copy(np.ascontiguousarray(self.current_cuboid[self.halo_size:2*self.halo_size, :, :]))

        [recvbuffer_1, recvbuffer_2] = self.halo_update_non_blocking(send_buffers=[sendbuffer_1, sendbuffer_2], neighbor_indices=[self.TOP, self.BOTTOM])

        if self.neighbors[self.TOP] != -2:
            self.current_cuboid[-self.halo_size:, :, :] = recvbuffer_2
        if self.neighbors[self.BOTTOM] != -2:
            self.current_cuboid[0:self.halo_size, :, :] = recvbuffer_1

        if self.is_tensor_mesh:
            return tf.convert_to_tensor(self.current_cuboid.reshape(1, self.sub_nx, self.sub_ny, self.sub_nz, 1))

        return self.current_cuboid

    # TODO: extract non-blocking p2p communications here
    def halo_update_non_blocking(self, send_buffers, neighbor_indices):
        updated_buffers = []
        recv_buffer2 = np.zeros_like(send_buffers[0])
        recv_buffer1 = np.zeros_like(send_buffers[1])
        buffer_size = send_buffers[0].shape
        if np.allclose(buffer_size, 1):
            self.data_type = MPI.FLOAT

        # non-blocking module of mpi4py
        requests = []
        requests.append(self.comm.Isend([send_buffers[0], self.data_type], dest=self.neighbors[neighbor_indices[0]]))
        requests.append(self.comm.Isend([send_buffers[1], self.data_type], dest=self.neighbors[neighbor_indices[1]]))
        requests.append(self.comm.Irecv([recv_buffer2, self.data_type], source=self.neighbors[neighbor_indices[1]]))
        requests.append(self.comm.Irecv([recv_buffer1, self.data_type], source=self.neighbors[neighbor_indices[0]]))
        MPI.Request.Waitall(requests)
        requests.clear()

        updated_buffers.append(recv_buffer2)
        updated_buffers.append(recv_buffer1)

        return updated_buffers

    ########## static methods which call without a instance of this class ##########
    @staticmethod
    def id_2_idx(rank, cols):
        """convert process rank to cartesian grid coordiantes

        Args:
            rank (int): process rank
            cols (int): length of x coordinates of grid coordiantes

        Returns:
            int,int: 2D cartesian grid coordinate (x,y)
        """
        return rank/cols, rank % cols

    @staticmethod
    def idx_2_id(rows, cols, id_row, id_col):
        """convert 2D cartesian grid coordinates to process rank

        Args:
            rows (int): Y
            cols (int): X
            id_row (int): y
            id_col (int): x

        Returns:
            int: process rank
        """
        if id_row >= rows or id_row < 0:
            return -1
        if id_col >= cols or id_col < 0:
            return -1
        return id_row * id_col + id_col

    @staticmethod
    def generate_proc_dim_2D(num_process):
        """generate 2D cartesian grid coordinate by number of processors

        Args:
            num_process (int): number of processors

        Returns:
            int,int: X,Y
        """
        rows, cols = 0, 0
        min_gap = num_process
        max_val = int(num_process**0.5 + 1)
        for i in range(1, max_val+1):
            if num_process % i == 0:
                gap = abs(num_process/i - i)
                if gap < min_gap:
                    min_gap = gap
                    rows = i
                    cols = int(num_process / i)
                    
        if rows == 1 or cols==1:
            return (-1,-1)

        return (rows, cols)

    @staticmethod
    def generate_proc_dim_3D(num_process):
        """Generate 3D cartesian topology dimensionality

        Args:
            num_process (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert num_process >= 1, f'The number of processors should be greater or equal to 1'
        left = 1
        right = num_process
        while right - left > 1e-5:
            mid = left + (right - left)//2
            cube_val = mid ** 3
            if cube_val == num_process:
                return (int(mid), int(mid), int(mid))
            elif cube_val > num_process:
                right = mid - 0.01
            elif cube_val < num_process:
                left = mid + 0.01

        if left**3 != num_process:
            return (num_process,1,1)

        return (left, left, left)

    @staticmethod
    def domain_decomposition_strip(mesh, num_process):
        """strip decomposition for 2D block structured mesh

        Args:
            mesh (numpy array): problem mesh
            num_process (int): number of processors

        Returns:
            list of numpy: divided sub-domains 
        """
        # print(num_process)
        sub_domains = np.hsplit(
            mesh, num_process)  # split the domain horizontally
        return sub_domains

    @staticmethod
    def domain_decomposition_grid(mesh, proc_grid_dim):
        """grid decomposition for 2D block-structured meshes

        Args:
            mesh (numpy array): problem mesh
            rows (int): X
            cols (int): Y

        Returns:
            list of numpy: sub-domains
        """
        nx, ny = mesh.shape
        assert nx % proc_grid_dim[0] == 0, f"{nx} rows is not evenly divisible by {proc_grid_dim[0]}"
        assert ny % proc_grid_dim[1] == 0, f"{ny} cols is not evenly divisible by {proc_grid_dim[1]}"
        sub_nx = nx//proc_grid_dim[0]
        sub_ny = ny//proc_grid_dim[1]
        return (mesh.reshape(nx//sub_nx, sub_nx, -1, sub_ny)
                .swapaxes(1, 2)
                .reshape(-1, sub_nx, sub_ny))

    @staticmethod
    def domain_decomposition_cube(mesh, proc_grid_dim):
        """grid decomposition for 3D block structured mesh

        Args:
            mesh (numpy array): problem mesh
            proc_grid_dim (tuple): (Z,X,Y)

        Returns:
            list of numpy: sub-domains
        """
        nx, ny, nz = mesh.shape
        
        assert nx % proc_grid_dim[0] == 0, f"{nx} grids along x axis is not evenly divisible by {proc_grid_dim[0]}"
        assert ny % proc_grid_dim[1] == 0, f"{ny} grids along y axis is not evenly divisible by {proc_grid_dim[1]}"
        assert nz % proc_grid_dim[2] == 0, f"{nz} grids along z axis is not evenly divisible by {proc_grid_dim[2]}"

        sub_nx = nx // proc_grid_dim[0]
        sub_ny = ny // proc_grid_dim[1]
        sub_nz = nz // proc_grid_dim[2]

        new_shape = (sub_nx, sub_ny, sub_nz)
        num_cubes = np.array(mesh.shape) // new_shape
        split_shape = np.column_stack([num_cubes, new_shape]).reshape(-1)
        order = np.array([0, 2, 4, 1, 3, 5])

        # return a numpy array
        return mesh.reshape(split_shape).transpose(order).reshape(-1, *new_shape)

    @staticmethod
    def padding_block_halo_1D(sub_domain, halo_size, halo_val=0):
        """padding the 1D subdomain with halos manually

        Args:
            sub_domain (numpy array): 1D sub-domain
            halo_size (int): width of the halo grids
            halo_val (int, optional): values to fill into the halo grids. Defaults to 0.

        Returns:
            numpy array: sub-domain with paddings
        """
        if tf.is_tensor(sub_domain):
            sub_domain = sub_domain.numpy()

        if sub_domain.ndim > 1:
            sub_domain = np.squeeze(sub_domain, axis=0)
            sub_domain = np.squeeze(sub_domain, axis=-1)

        return np.pad(sub_domain, (halo_size, halo_size), 'constant', constant_values=(halo_val,))

    @staticmethod
    def padding_block_halo_2D(sub_domain, halo_size, halo_val=0):
        """padding the 2D subdomain with halos manually

        Args:
            sub_domain (numpy array): 2D sub-domain
            halo_size (int): width of the halo grids
            halo_val (int, optional): values to fill into the halo grids. Defaults to 0.

        Returns:
            numpy array: sub-domains with paddings
        """
        if tf.is_tensor(sub_domain):
            sub_domain = sub_domain.numpy()

        if sub_domain.ndim > 2:
            sub_domain = np.squeeze(sub_domain)

        # note padding halo values with 0 by default
        return np.pad(sub_domain, (halo_size, halo_size), 'constant', constant_values=(halo_val,))

    @staticmethod
    def padding_block_halo_3D(sub_cube, halo_size, halo_val=0):
        """padding the 3D subdomain with halos manually

        Args:
            sub_cube (numpy array): 3D sub-domain
            halo_size (int): width of the halo grids
            halo_val (int, optional): values to fill into the halos. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if tf.is_tensor(sub_cube):
            sub_cube = sub_cube.numpy()

        if sub_cube.ndim > 3:
            sub_cube = HaloExchange.remove_one_dims(sub_cube,expected_dim=3)

        # note padding halo values with 0 by default
        return np.pad(sub_cube, (halo_size, halo_size), 'constant', constant_values=(halo_val,))

    @staticmethod
    def remove_one_dims(mesh,expected_dim):
        """remove the trivial 1-dimensions from the tensor object

        Args:
            input (numpy array): numpy array that converted from the tensor

        Returns:
            np.array: the squeezed numpy array
        """
        while mesh.ndim != expected_dim:
            if mesh.shape[0] == 1 and mesh.shape[-1] == 1:
                mesh = np.squeeze(mesh, axis=0)
                mesh = np.squeeze(mesh, axis=-1)
            else:
                break
        return mesh

############################## NEW MODIFICATIONS BELOW ##############################

def halo_update_one_sided_comm():
    #TODO: other communication patterns
    return NotImplementedError()
