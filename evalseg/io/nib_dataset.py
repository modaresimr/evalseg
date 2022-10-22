from .dataset import Dataset, get_file, CT, GT, PREDS
from .nib import read_nib


class NibDataset(Dataset):

    def __init__(self, path):
        super().__init__(path)

    def get_CT(self, id):
        return read_nib(get_file(self.dataset_info[CT][f'{id}']))

    def get_groundtruth(self, id):
        return read_nib(get_file(self.dataset_info[GT][f'{id}']))

    def get_prediction(self, id,method):
        return read_nib(get_file(self.dataset_info[PREDS][method][f'{id}']))


