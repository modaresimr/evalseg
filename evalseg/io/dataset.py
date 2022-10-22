import json
import os
import zipfile
from .. import common
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

        self.dataset_info = load_dataset_info(path)

    def get(self, typ, case):
        if case == self._cache_case and self._cache.get(typ, None) is not None:
            return self._cache[typ]

        if typ == CT:
            path = self.dataset_info[CT][f"{case}"]
        elif typ == GT:
            path = self.dataset_info[GT][f"{case}"]
        else:
            path = self.dataset_info[PREDS][typ][f"{case}"]

        return self.read_file(path)

    def read_file(self, path):
        return get_file(path)

    def get_CT(self, id):
        return self.get(CT, id)

    def get_groundtruth(self, id):
        return self.get(GT, id)

    def get_prediction(self, method, id):
        return self.get(method, id)

    def get_prediction_methods(self):
        return list(self.dataset_info[PREDS].keys())

    def load_all_of_case(self, case, load_ct=True, load_gt=True, load_preds=None):
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

    def clear_case():
        del _cache

    def _load(self, inp):
        typ = inp[0]
        case = inp[1]
        return self.get(typ, case)

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
            current[t] = f"{path}#{zpath}"

    return res


def _load_dir_info(path):
    res = {}
    for file in os.listdir(path):
        res[file.replace('.zip', '')] = load_dataset_info(f"{path}/{file}")
    return res


def load_dataset_info(path):
    if os.path.isfile(path) and ".zip" in path:
        return _load_zip_info(path)
    elif os.path.isdir(path):
        return _load_dir_info(path)
    else:
        return path


def get_file(path):
    try:
        if "#" in path:
            zname, zpath = path.split("#")
            with zipfile.ZipFile(zname, "r") as archive:
                return archive.read(zpath)
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"can not load {path}: {e}")
        return None
