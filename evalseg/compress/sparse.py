import numpy as np
import sparse


disable = True


def compress_arr(arr):
    if disable:
        return arr
    if type(arr) == dict:
        return {p: compress_arr(arr[p]) for p in arr}

    if isinstance(arr, np.ndarray):
        from collections import Counter
        arr_1d = arr.reshape(-1)

        fill, count = Counter(arr_1d).most_common(1)[0]

        if count/len(arr_1d) > 0.1:
            return sparse.COO.from_numpy(arr, fill_value=fill)

    return arr


def decompress_arr(arr):
    if disable:
        return arr
    if type(arr) == dict:
        return {p: decompress_arr(arr[p]) for p in arr}
    if type(arr) == sparse.COO:
        return arr.todense()
    return arr
