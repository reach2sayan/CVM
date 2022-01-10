from scipy.optimize import Bounds

class CorrBounds:
    """
    Class to constrain the trial correlations for local search
    """

    def __init__(self, num_clusters):

        self.num_clusters = num_clusters

    def get_pd_bounds(self, FIXED_CORR_1):

        bounds_corrs = Bounds([1, FIXED_CORR_1, *[-1]*(self.num_clusters-2)],
                              [1, FIXED_CORR_1, *[1]*(self.num_clusters-2)],
                             )
        return bounds_corrs

    def get_corrscan_bounds(self, FIXED_CORR_1, FIXED_CORR_2):

        bounds_corrs = Bounds([1, FIXED_CORR_1, FIXED_CORR_2, *[-1]*(self.num_clusters-3)],
                              [1, FIXED_CORR_1, FIXED_CORR_2, *[1]*(self.num_clusters-3)],
                              keep_feasible=True
                             )
        return bounds_corrs

    def get_muscan_bounds(self):

        bounds_corrs = Bounds([1, *[-1]*(self.num_clusters-1)],
                              [1, *[1]*(self.num_clusters-1)],
                              keep_feasible=True
                             )
        return bounds_corrs
