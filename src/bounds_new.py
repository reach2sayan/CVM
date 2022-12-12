from __future__ import annotations
from dataclasses import dataclass
from scipy.optimize import Bounds
import numpy as np

@dataclass
class CorrBounds:
    """
    Class to constrain the bounds of the allowed correlations
    """
    _num_clusters: int
    _num_single_clusters: int
    _FIXED_CORRS: np.ndarray

    @property
    def sro_bounds(self):

        lower_bound = np.array(
            [1, *self._FIXED_CORRS, *[-1]*(self._num_clusters-1-self._num_single_clusters)])
        upper_bound = np.array(
            [1, *self._FIXED_CORRS, *[1]*(self._num_clusters-1-self._num_single_clusters)])
        return Bounds(lower_bound, upper_bound)
