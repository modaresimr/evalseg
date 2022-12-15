from ..metrics import MME, MultiClassMetric, MultiMetric, HD, Voxel, NSD
from ..io import Dataset
from ..common import NumpyEncoder
import numpy as np
import pandas as pd
from ipywidgets import interact, interact_manual, IntSlider
import os
import glob
import json
import psutil


def get_metrics(num_classes) -> MultiMetric:
    metrics = {
        'mme': MME(),
        'hd': HD(),
        'voxel': Voxel(),
        'nsd t=1': NSD(tau=1),
        'nsd t=5': NSD(tau=5),

        # evalseg.metrics.BD,
    }

    return MultiMetric(metrics_dic={m: MultiClassMetric(metrics[m], num_classes)for m in metrics})


def measure_all_metrics_for_all_pred(dataset: Dataset, case, parallel=1, max_cpu=1):
    gto = dataset.get_groundtruth(case)
    metric = get_metrics(dataset.num_labels)
    metric.set_reference(gto)

    preds = {p: dataset.get(p, case) for p in dataset.get_prediction_methods()}

    res = metric.evaluate_multi(preds, parallel=parallel, max_cpu=max_cpu)
    return res


def mutli_run_all_datasets(root_data, out_root):
    root_data = 'datasets'
    out_root = 'out'
    os.makedirs(out_root, exist_ok=True)

    for dataset_name in sorted([d for d in os.listdir(root_data) if os.path.isdir(f'{root_data}/{d}')]):
        mutli_run_dataset(root_data, dataset_name, out_root)


def mutli_run_dataset(root_data, dataset_name, out_root):  # pragma: no cover
    """
    The main function executes on commands:
    `python -m evalseg` and `$ evalseg `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    print("Exec")

    dataset = Dataset(f'{root_data}/{dataset_name}')

    for case in sorted(dataset.get_available_ids()):
        out_path = f'{out_root}/{dataset_name}-{case}.json'
        print(out_path)
        if os.path.isfile(f'{out_path}'):
            continue
        maxcpu = psutil.virtual_memory().available//(50 * 1024 * 1024 * 1024)+1

        res = measure_all_metrics_for_all_pred(dataset, case, max_cpu=maxcpu)

        with open(f'{out_path}', 'w') as f:
            json.dump(res, f, indent=4, cls=NumpyEncoder)
