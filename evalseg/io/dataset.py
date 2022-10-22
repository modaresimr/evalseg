import json
import os
import zipfile

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

    def get_CT(self, id):
        return get_file(self.dataset_info[CT][f"{id}"])

    def get_groundtruth(self, id):
        return get_file(self.dataset_info[GT][f"{id}"])

    def get_prediction(self, method, id):
        return get_file(self.dataset_info[PREDS][method][f"{id}"])

    def get_prediction_methods(self):
        return list(self.dataset_info[PREDS].keys())

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
