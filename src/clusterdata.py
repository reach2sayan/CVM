import dataloader as dl
import numpy as np

class ClusterInfo:

    def __init__(self,clusters_fname, kb_fname, configcoef_fname, config_fname, vmat_fname, eci_fname):

        self.clusters = dl.read_clusters(clusters_fname)
        self.kb = dl.read_kbcoeffs(kb_fname)
        self.configcoef = dl.read_configcoef(configcoef_fname)
        self.configs = dl.read_configs(config_fname)
        self.vmat = dl.read_vmatrix(vmat_fname)
        self.eci = dl.read_eci(eci_fname)

    def __repr__(self):

        print("ECI:")
        print(self.eci)
        print("KB:")
        print(self.kb)
        print("Vmatrix")
        print(self.vmat)
        print("Clusters:")
        print(self.clusters)
        print("Cluster Coefficients:")
        print(self.configcoef)
        return ''

    def rho(self,corrs):
        for cluster_idx, _ in self.clusters.items():
            print(np.matmul(self.vmat[cluster_idx],corrs))
        return ''

    def check_result_validity(self,corrs):
        try:
            for cluster_idx in self.clusters.keys():
                assert np.isclose(np.inner(self.configcoef[cluster_idx], np.matmul(self.vmat[cluster_idx], corrs)), 1.0)
        except AssertionError:
            print("Failed to find solution...Current Rho")
            print(f'Rho: \n {self.rho(corrs)}')
            return 






