from scipy.optimize import Bounds
import numpy as np


class CorrBounds:
    """
    Class to constrain the bounds of the allowed correlations
    """

    def __init__(self, num_clusters, num_single_clusters):
        self.num_clusters = num_clusters
        self.num_single_clusters = num_single_clusters

    def get_sro_bounds(self, FIXED_CORRS):

        lower_bound = np.array(
            [1, *FIXED_CORRS, *[-1]*(self.num_clusters-1-self.num_single_clusters)])
        upper_bound = np.array(
            [1, *FIXED_CORRS, *[1]*(self.num_clusters-1-self.num_single_clusters)])
        sro_bounds = Bounds(lower_bound, upper_bound,)

        return sro_bounds
