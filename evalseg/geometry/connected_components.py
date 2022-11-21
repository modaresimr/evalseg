
from multiprocessing import Process, Queue
import cc3d
import numpy as np
import concurrent.futures


def connected_components(bianry_array, return_N=True):
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     f = executor.submit(__cc3d_connected_components, bianry_array)
    #     ret, count = f.result()
    ret, count = __cc3d_connected_components(bianry_array)
    if return_N:
        return ret, count
    return ret


def __cc3d_connected_components(bianry_array):
    ri = tuple([np.s_[0:bianry_array.shape[i]] for i in range(bianry_array.ndim)])
    if any([bianry_array.shape[i] < 2 for i in range(bianry_array.ndim)]):
        ri_new_shape = tuple([bianry_array.shape[i]+2 for i in range(bianry_array.ndim)])
        ri = tuple([np.s_[1:bianry_array.shape[i]+1] for i in range(bianry_array.ndim)])
        bianry_array2 = np.zeros(ri_new_shape, bianry_array.dtype)
        bianry_array2[ri] = bianry_array
        bianry_array = bianry_array2

    labels, count = cc3d.connected_components(bianry_array, return_N=True)

    return labels[ri], count
