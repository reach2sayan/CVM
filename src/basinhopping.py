import numpy as np
from scan_disordered import get_random_structure
import subprocess

def basin_hopping_callback(corrs, F, accept):
    """
    Function to print diagnostic while fitting.
    Not being used right now
    """
    if accept:
        print(f'Current Optimised Free Energy: {F}')
        print(f'Current Minimum COrrelation: {corrs}')
        print("===========================")

#TODO Add a proper callback
class BasinHoppingCallback:
    """
    Class to print output from bassinhoppinh
    """
    def __init__(self, cluster_data, F,):
        pass


class BasinHoppingBounds:
    """
    Class to constrain the trial correlations of Basin Hopping
    """
    def __init__(self,all_vmat):
        self.all_vmat = all_vmat

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        rho = self.all_vmat @ x
        return np.all((rho >= 0.0) & (rho <= 1.0))

class BasinHoppingStep:

    def __init__(self, cluster_data, structure, lattice_file, clusters_file, trial_variance=0.01, seed=42, random_trial=False):

        self.structure = structure,
        self.lattice_file = lattice_file,
        self.clusters_file = clusters_file,
        self.trial_variance = trial_variance
        self.cluster_data = cluster_data
        self.rng = np.random.default_rng(seed)

        if random_trial:
            self.random_trial = True
            self.randgen = get_random_structure(structure)
        else:
            self.stepsize = trial_variance
            self.randgen = None
            self.random_trial = False

    def __call__(self, x):

        if self.random_trial:
            next(self.randgen)
            corrs_attempt = subprocess.run(['corrdump',
                                            '-c',
                                            f'-cf={self.structure[0]}/{self.clusters_file[0]}',
                                            f'-s={self.structure[0]}/randstr.in',
                                            f'-l={self.structure[0]}/{self.lattice_file[0]}'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           check=True
                                          )
            # convert from bytes to string list
            corrs_attempt = corrs_attempt.stdout.decode('utf-8').split('\t')[:-1]
            corrs_attempt = np.array(corrs_attempt, dtype=np.float32)  # convert to arrays
            assert self.cluster_data.check_result_validity(corrs_attempt)
            print(corrs_attempt)

            return corrs_attempt

        jitter = np.array([0,
                           *[0] *
                           len(self.cluster_data.single_point_clusters),
                           *self.rng.normal(0,
                                            self.stepsize,
                                            self.cluster_data.num_clusters -
                                            len(self.cluster_data.single_point_clusters) - 1
                                           )
                          ]
                         )
        return x + jitter
