from pathlib import Path
import numpy as np
import json
import os
import zipfile
import gzip
from .. import common
from . import SegmentArray
from .nib import read_nib
import pickle
GT = "GroundTruth"
CT = "CT"
PREDS = "Predictions"


class Dataset:
    def __init__(self, path):
        try:
            with open(f"{path}/metadata.json", 'r') as f:
                self.labels = json.load(f)["labels"]
        except Exception as e:
            print(f"""error in metadata.json {e} -->
                    consider binary labels: 0->background, 1->forground""")
            self.labels = {"0": "background", "1": "forground"}
        self.labels = {int(k): self.labels[k] for k in self.labels}
        self.num_labels = max(self.labels.keys())+1

        self.dataset_info = load_dataset_info(path)
        self._cache_case = None
        self._cache = {}

    def _is_valid_cache(self, typ, case):
        if case != self._cache_case:
            return False
        if not hasattr(self, '_cache'):
            return False
        if self._cache.get(typ, None) is None:
            return False
        return True

    def get(self, typ, case, numpy_array=False):
        if self._is_valid_cache(typ, case):
            return self._cache[typ]

        if typ == CT:
            path = self.dataset_info[CT].get(f"{case}", None)
        elif typ == GT:
            path = self.dataset_info[GT].get(f"{case}", None)
        else:
            path = self.dataset_info[PREDS][typ].get(f"{case}", None)
        if path == None:
            print(f"can not load {typ} {case}")
            return None, None
        data = self.read_file(path)

        if numpy_array:
            if isinstance(data, SegmentArray):
                return data.todense(), data.voxelsize
            else:
                return data

        if isinstance(data, SegmentArray):
            return data

        data_arr, voxelsize = data
        return SegmentArray(data_arr, voxelsize, multi_part=typ != CT)
        # if typ != CT and type(data) == tuple and type(data[0]) == np.ndarray:
        #     data_arr, voxelsize = data
        #     return MultiClassSegment(data_arr, voxelsize), voxelsize
        # return data

    def read_file(self, path):
        return get_file(path)

    def get_CT(self, id) -> SegmentArray:
        return self.get(CT, id)

    def get_groundtruth(self, id) -> SegmentArray:
        return self.get(GT, id)

    def get_prediction(self, method, id) -> SegmentArray:
        return self.get(method, id)

    def get_prediction_methods(self, case=None):
        all_preds = list(self.dataset_info[PREDS].keys())
        if case:
            all_preds = [p for p in all_preds if case in self.dataset_info[PREDS][p]]
        return all_preds

    def load_all_of_case(self, case, load_ct=True, load_gt=True, load_preds=None):
        self.clear_case()
        methods = []
        if load_ct:
            methods.append(CT)
        if load_gt:
            methods.append(GT)
        if load_preds is None:
            load_preds = self.get_prediction_methods()

        methods = [*methods, *load_preds]

        methods = [(k, case) for k in methods]
        self._cache = {k[0]: v for k, v in common.parallel_runner(self._load, methods)}
        self._cache_case = case

    def clear_case(self):
        if hasattr(self, '_cache'):
            del self._cache

    def _load(self, inp):
        typ = inp[0]
        case = inp[1]

        return self.get(typ, case, compress=True)

    def get_available_ids(self):
        res = {}
        for m in self.dataset_info[PREDS]:
            for i in self.dataset_info[PREDS][m]:
                res[i] = res.get(i, 0) + 1

        for d in [GT]:
            res = {i: res[i] for i in res if i in self.dataset_info[d]}

        return [i for i in res]

    def get_all_ids(self):
        return [i for i in self.dataset_info[GT]]


def _load_zip_info(path):
    res = {}
    with zipfile.ZipFile(path, "r") as archive:
        for zpath in archive.namelist():
            if zpath.startswith(".") or "/." in zpath or zpath.endswith("/"):
                continue

            h, t = os.path.split(zpath)

            current = res
            if h != '':
                for c in h.split("/"):
                    if not (c in current):
                        current[c] = {}
                    current = current[c]
            current[t.split('.')[0]] = f"{path}#{zpath}"

    return res


def _load_dir_info(path):
    res = {}
    for file in os.listdir(path):
        key = file.split('.')[0]
        value = load_dataset_info(f"{path}/{file}")
        # if key not in res:
        res[key] = value
    return res


def load_dataset_info(path):
    if os.path.isfile(path) and ".zip" in path:
        return _load_zip_info(path)
    elif os.path.isdir(path):
        return _load_dir_info(path)
    else:
        return path


def get_file(path: str):
    try:
        if "#" in path:
            zname, zpath = path.split("#")
            zdir, zfile = os.path.split(zpath)
            zdir = f'{zdir}/' if zdir != '' else ''
            fpath = f'{zdir}{zfile.split(".")[0]}'
            with zipfile.ZipFile(zname, "r") as archive:
                for eext in ['.gz', '']:
                    for ext in ['pkl', 'nii']:
                        try:
                            f = archive.read(f'{fpath}.{ext}{eext}')
                            if eext == '.gz':
                                return open_data(gzip.decompress(f), ext)
                            return open_data(f, ext)
                        except:
                            continue
                raise Exception('not found')

        dir, file = os.path.split(path)
        dir = f'{dir}/' if dir != '' else ''
        fpath = f'{dir}/{file.split(".")[0]}'
        for eext in ['.gz', '']:
            for ext in ['pkl', 'nii']:
                try:
                    if eext == '.gz':
                        with gzip.open(f'{fpath}.{ext}{eext}') as f:
                            return open_data(f.read(), ext)
                    else:
                        with open(f'{fpath}.{ext}{eext}', "rb") as f:
                            return open_data(f.read(), ext)
                except:
                    continue
        raise Exception('not found')

    except Exception as e:
        print(f"can not load {path}: {e}")
        raise
        return None


def open_data(data, typ):
    if 'nii' in typ:
        return read_nib(data)
    if 'pkl' in typ:
        return pickle.loads(data)

    raise Exception('unsupported data type')
