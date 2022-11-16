# from . import MetricABS
# import numpy as np
# from . import Voxel


# class VS(MetricABS):
#     def __init__(self, debug={}):
#         super().__init__(debug)
#         self.voxelmetric = Voxel()
#         pass

#     def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
#         super().set_reference(reference, spacing, **kwargs)
#         self.voxelmetric.set_reference(reference, spacing, **kwargs)

#     def evaluate(self, test: np.ndarray, **kwargs):
#         is2d = (test.ndim == 2 or test.shape[2] == 1)

#         cm = self.voxelmetric.evaluate(test)
#         tp, fp, fn = cm['tp'], cm['fp'], cm['fn']
#         # Compute VS
#         if (2*tp + fp + fn) != 0:
#             vs = 1 - (np.abs(fn-fp) / (2*tp + fp + fn))
#         else:
#             vs = 1.0 - 0.0
#         # Return VS score
#         return vs
