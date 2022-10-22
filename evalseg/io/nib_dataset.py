from .dataset import Dataset, get_file, CT, GT, PREDS
from .nib import read_nib


class NibDataset(Dataset):

    def __init__(self, path):
        super().__init__(path)

    def read_file(self, path):
        return read_nib(super().read_file(path))
