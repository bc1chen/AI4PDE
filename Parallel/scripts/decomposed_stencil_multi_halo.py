
from time import perf_counter
from pprint import pprint
import numpy as np
from halos_exchange import HaloExchange

start = perf_counter()
h = HaloExchange(structured=True, halo_size=3, tensor_used=False,
                 double_precision=True, corner_exchanged=True) # here we try the halo size with 3

mesh = np.array([[0.78261493, 0.97090189, 0.66327036, 0.76848217, 0.42800353,0.77528106],
                 [0.56767283, 0.57780701, 0.33225302, 0.66295884, 0.87718818,0.56417634],
                 [0.54046498, 0.17588775, 0.71082921, 0.30586232, 0.34552216,0.92550091],
                 [0.78670413, 0.32494218, 0.53005155, 0.79372642, 0.0620199,0.57615613],
                 [0.9406587, 0.83912257, 0.33540851, 0.25028467, 0.84222929,0.52955617],
                 [0.79936637, 0.374291, 0.6705437, 0.84339835, 0.32246901,0.70702548]])
# pprint(mesh)
x, y, sub_domain = h.initialization(mesh,is_periodic=False,is_reordered=False)
print(sub_domain.shape)
print(sub_domain)

sub_domain = h.structured_halo_update_2D(sub_domain)
# print(sub_domain)

np.save('parallel_out/mult_halo_{}'.format(h.rank),sub_domain)

h.clear_cache()
end = perf_counter()
print(end - start)
