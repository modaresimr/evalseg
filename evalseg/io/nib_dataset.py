# from .dataset import Dataset, get_file, CT, GT, PREDS
# from .nib import read_nib
# import numpy as np
# from ..compress import compress_arr, decompress_arr


# class NibDataset(Dataset):

#     def __init__(self, path):
#         super().__init__(path)

#     def get(self, typ, case, compress=False):
#         data = super().get(typ, case, compress)
#         if compress:
#             return compress_arr(data[0]), data[1]
#         else:
#             return decompress_arr(data[0]), data[1]

#     def read_file(self, path):
#         return read_nib(super().read_file(path))
