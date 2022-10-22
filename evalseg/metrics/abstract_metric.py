from collections import defaultdict

import numpy as np

from ..common import Object

# pylint: disable-all
# pylint: disable=all
# pylint: skip-file


class MetricABS(Object):
    def __init__(self, num_classes: int, debug={}):
        self.num_classes = num_classes
        self.debug = defaultdict(str, debug)
        pass

    def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        self.reference = reference
        self.spacing = np.array(
            spacing if not (spacing is None) else [1, 1, 1]
        )
        self.helper = self.calculate_info(
            reference, self.spacing, self.num_classes, **kwargs
        )

    def calculate_info(
        cls,
        reference: np.ndarray,
        spacing: np.ndarray = None,
        num_classes: int = 2,
        **kwargs
    ):
        pass

    def evaluate_single(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        spacing: np.ndarray = None,
        **kwargs
    ):
        self.set_reference(reference, spacing, **kwargs)
        return self.evaluate(test, **kwargs)

    def evaluate(self, test: np.ndarray, **kwargs):
        pass
