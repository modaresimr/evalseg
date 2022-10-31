from .dataset import Dataset, get_file, CT, GT, PREDS
from .nib import read_nib
import numpy as np
import sparse


class NibDataset(Dataset):

    def __init__(self, path):
        super().__init__(path)

    def get(self, typ, case, compress=False):
        data = super().get(typ, case, compress)
        if not compress and type(data[0]) == sparse.COO:
            return data[0].todense(), data[1]
        elif compress:
            return compress_numpy(data[0]), data[1]
        return data

    def read_file(self, path):
        return read_nib(super().read_file(path))

    


def compress_numpy(arr):

    if np.sum(arr == 0)/np.sum(arr == arr) > 0.3:
        return sparse.COO.from_numpy(arr)
    return arr
