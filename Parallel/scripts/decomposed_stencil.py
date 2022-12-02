from halo_exchange_upgraded import HaloExchange
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from time import perf_counter


mesh = np.array([[0.36957142, 0.35448541, 0.87534055, 0.41334616, 0.58921851,
                  0.33873342, 0.84686359, 0.70661254],
                 [0.70057196, 0.88576827, 0.16987487, 0.84091359, 0.33203134,
                  0.98150579, 0.25927917, 0.52427803],
                 [0.46945511, 0.71938923, 0.92161953, 0.47165736, 0.75622128,
                  0.54595177, 0.30856705, 0.56784142],
                 [0.09464055, 0.34605817, 0.02577937, 0.00983891, 0.30026129,
                  0.1449963, 0.17470095, 0.91703354],
                 [0.50243758, 0.82406356, 0.22732425, 0.25323539, 0.71764112,
                  0.88660062, 0.13041651, 0.76067969],
                 [0.74190659, 0.42878908, 0.14753486, 0.75686801, 0.56198465,
                  0.09739528, 0.44083451, 0.11082951],
                 [0.34638461, 0.95217675, 0.08757205, 0.49765066, 0.89150327,
                  0.70783763, 0.19393284, 0.77442823],
                 [0.33802426, 0.63746115, 0.14900671, 0.30016339, 0.43270171,
                  0.11015575, 0.73895399, 0.23091873]])

he = HaloExchange(structured=True, tensor_used=False,
                  double_precision=True, corner_exchanged=True)
sub_x, sub_y, current_domain = he.initialization(
    mesh, is_periodic=False, is_reordered=False)

current_domain = he.structured_halo_update_2D(current_domain)

# print(f'{sub_nx},{sub_ny}')
# pprint(current_domain)

# mesh = np.array([[9, 6, 3, 7, 8, 8, 5, 9],
#                  [7, 8, 9, 8, 2, 2, 9, 3],
#                  [6, 3, 7, 5, 4, 5, 9, 8],
#                  [1, 5, 7, 3, 2, 8, 1, 6],
#                  [6, 2, 7, 3, 3, 5, 9, 7],
#                  [8, 8, 7, 1, 1, 4, 2, 4],
#                  [5, 2, 5, 1, 5, 1, 5, 9],
#                  [6, 4, 7, 6, 5, 1, 1, 9]])


rank = he.rank
current_domain_new = np.zeros_like(current_domain)

start = perf_counter()
for t in range(1000):
    np.save('parallel_steps/parallel_{}_{}'.format(rank, t), current_domain)
    for i in range(1, sub_x+1):
        for j in range(1, sub_y+1):
            current_domain_new[i][j] = current_domain[i][j] + (current_domain[i-1][j] + current_domain[i+1]
                                                               [j] + current_domain[i][j-1] + current_domain[i][j+1])*0.25

    # np.save('parallel_steps/parallel_{}_{}'.format(rank,t),current_domain_new)
    # update the halo at each time step
    current_domain = np.copy(current_domain_new)
    current_domain = he.structured_halo_update_2D(current_domain)
    # np.save('parallel_steps/parallel_{}_{}'.format(rank,t),current_domain)


end = perf_counter()
print("[TOTAL TIME] {}".format(end - start))
np.save('parallel_out/ans_{}.npy'.format(rank), current_domain[1:-1, 1:-1])
