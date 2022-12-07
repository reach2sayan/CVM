import numpy as np
from scan_disordered import get_random_structure
import subprocess

class BasinHoppingCallback:
    """
    Class to print output from basinhopping
    """
    def __init__(self, cluster_data, T, num_atoms_per_clus, corr_rnd):

        self.cluster_data = cluster_data,
        self.T = T,
        self.num_atoms_per_clus = num_atoms_per_clus,
        self.corr_rnd = corr_rnd

    def __call__(self, x, f, accept):
        print(f'Current correlations: {x}')
#        print(f'Trial Validity: {self.cluster_data.check_result_validity(x)}')
        print(f'Current Free Energy @ T = {self.T[0]}K : {f/self.num_atoms_per_clus[0]}')
        print(f"Fully Disordered? : {np.allclose(x,self.corr_rnd)}")
        print(f"Accepted? : {accept}")
        print('\n====================================\n')

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

    def __init__(self,
                 e_F,
                 mults_eci,
                 multconfig_kb,
                 all_vmat,
                 vrhologrho,
                 temp,
                 cluster_data,
                 structure,
                 num_atoms_per_clus,
                 lattice_file,
                 clusters_file,
                 stepsize=0.01,
                 seed=42,
                 display_inter=False,
                 random_trial=False):

        self.structure = structure
        self.lattice_file = lattice_file
        self.clusters_file = clusters_file
        self.cluster_data = cluster_data
        self.rng = np.random.default_rng(seed)
        self.F = e_F
        self.mults_eci = mults_eci
        self.multconfig_kb = multconfig_kb
        self.all_vmat = all_vmat
        self.vrhologrho = vrhologrho
        self.temp = temp
        self.display_inter = display_inter
        self.num_atoms_per_clus = num_atoms_per_clus

        if random_trial:
            self.random_trial = True
            self.randgen = get_random_structure(structure)
        else:
            self.stepsize = stepsize
            self.randgen = None
            self.random_trial = False

        self.step_count = 0

    def __call__(self, x):

        if self.random_trial:
            next(self.randgen)
            corrs_attempt = subprocess.run(['corrdump',
                                            '-c',
                                            f'-cf={self.structure}/{self.clusters_file}',
                                            f'-s={self.structure}/randstr.in',
                                            f'-l={self.structure}/{self.lattice_file}'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           check=True
                                          )
            # convert from bytes to string list
            corrs_attempt = corrs_attempt.stdout.decode('utf-8').split('\t')[:-1]
            corrs_attempt = np.array(corrs_attempt, dtype=np.float32)  # convert to arrays
            assert self.cluster_data.check_result_validity(corrs_attempt)

        else:
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
            corrs_attempt = x + jitter
        fattempt = self.F(corrs_attempt,
                          self.mults_eci,
                          self.multconfig_kb,
                          self.all_vmat,
                          self.vrhologrho,
                          self.temp
                         )
        if self.display_inter:
            print(f'Trial No.: {self.step_count}')
            print(f'Current attempt correlations: {corrs_attempt}')
            print(f'Trial Validity: {self.cluster_data.check_result_validity(corrs_attempt)}')
            print(f'Attempt Free Energy @ T = {self.temp}K : {fattempt/self.num_atoms_per_clus}')
        
        self.step_count += 1
        return corrs_attempt
