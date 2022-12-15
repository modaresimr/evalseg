from . import MetricABS
import numpy as np
from . import Voxel


class BD(MetricABS):
    def __init__(self, debug={}):
        super().__init__(debug)
        self.voxelmetric = Voxel()
        pass

    