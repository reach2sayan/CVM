import numpy as np
import re
import sys


class ClusterInfo:
    """
    Class containing the clusters information.
    Input: Filenames containing
           1. No. of clusters,
           2. ECI
           3. Kikuchi-Baker Coefficients
           4. Config Coefficients
           5. Config INformation (this is not particularly used in calculations)
           6. V-Matrix (for correlations --> rhos)
           7. Cluster Only Flag (to avoid loding extra files if cluster variation is not required)
    Properties:
           1. Num of clusters
           2. Num of configs
           3. Num of single points correlations
    Class Methods: File parsers for every input file
    Object Methods:
           1. Gives the rhos for a particular set of correlation with current cluster description
           2. Method to check if the rhos sum ~ 1
    """

    def __init__(self, clusters_fname,
                 eci_fname,
                 kb_fname=None,
                 configcoef_fname=None,
                 config_fname=None,
                 vmat_fname=None,
                 cluster_only=False):

        try:
            if cluster_only:
                print('Creating an Object with cluster and ECI only. Not suitable for CVM.')
                self.clusters = ClusterInfo.read_clusters(clusters_fname)
                self.kb = None
                self.configcoef = None
                self.configs = None
                self.vmat = None
                self.eci = ClusterInfo.read_eci(eci_fname)
            else:
                self.clusters = ClusterInfo.read_clusters(clusters_fname)
                self.kb = ClusterInfo.read_kbcoeffs(kb_fname)
                self.configcoef = ClusterInfo.read_configcoef(configcoef_fname)
                self.configs = ClusterInfo.read_configs(config_fname)
                self.vmat = ClusterInfo.read_vmatrix(vmat_fname)
                self.eci = ClusterInfo.read_eci(eci_fname)
        except FileNotFoundError as fnfe:
            print('File Not Found for instanting cluster description. Exiting...')
            print(fnfe)
            sys.exit(1)

        try:
            assert len(self.eci) == len(self.clusters)
        except AssertionError:
            print(
                f'Number of ECIs ({len(self.eci)}) does not match number of clusters ({len(self.clusters)}). Exiting...')
            sys.exit(1)

    def get_rho(self, corrs):

        for config_idx in self.configcoef.keys():
            print(self.vmat[config_idx] @ corrs)
        return

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

    @property
    def num_configs(self):
        return len(self.configcoef)

    @property
    def single_point_clusters(self):
        return [cluster_idx for cluster_idx, cluster in self.clusters.items() if cluster['type'] == 1]

    @property
    def num_clusters(self):
        return len(self.clusters)

    def check_result_validity(self, corrs):
        try:
            for config_idx in self.configcoef:
                assert np.isclose(np.inner(self.configcoef[config_idx], np.matmul(
                    self.vmat[config_idx], corrs)), 1.0, rtol=1e-3)
        except AssertionError:
            print("Invalid Rho")
            return False

        return True

    @classmethod
    def read_clusters(cls, clusters_fname):

        clusters = {}

        with open(clusters_fname, 'r') as fclusters:
            # Read blocks separated by 1 empty line
            temp_clusters = fclusters.read().split('\n\n')

        for idx, cluster in enumerate(temp_clusters):
            if cluster == '':
                continue
            line = cluster.split('\n')  # If not empty split by lines
            multiplicity = int(line[0])  # 1st line
            length = float(line[1])  # largest distance between two atoms
            num_points = int(line[2])  # type of cluster
            clusters[idx] = {'mult': multiplicity,
                             'length': length,
                             'type': num_points
                             }

        return clusters

    @classmethod
    def read_kbcoeffs(cls, kb_fname):

        kb = {}
        fkb = open(kb_fname, 'r')
        _ = next(fkb)  # ignore first line

        temp_kb = fkb.read()
        temp_kb = temp_kb.split('\n')  # split file linewise

        for idx, kbcoeff in enumerate(temp_kb):
            if kbcoeff == '':  # check for spurious empty blocks
                continue
            kb[idx] = float(kbcoeff)

        fkb.close()
        return kb

    @classmethod
    def read_configcoef(cls, configcoef_fname):

        configcoef = {}
        pattern1 = re.compile("\n\n\n")
        pattern2 = re.compile("\n\n")

        with open(configcoef_fname, 'r') as fsubmult:
            _ = next(fsubmult)  # ignore first line
            temp_submult = fsubmult.read()
            # split lines into blocks separated by 2 empty lines
            temp_submult = pattern2.split(temp_submult)

        for idx, submult in enumerate(temp_submult[:-1]):
            submult = submult.split('\n')  # split into number of subclusters
            while("" in submult):
                submult.remove("")  # remove empty blocks
            # also ignore 1st line of each block
            configcoef[idx] = list(map(float, submult[1:]))

        return configcoef

    @classmethod
    def read_vmatrix(cls, vmat_fname):

        pattern1 = re.compile("\n\n\n")
        pattern2 = re.compile("\n\n")

        vmat = {}
        with open(vmat_fname, 'r') as fvmat:
            _ = next(fvmat)  # ignore first lie
            temp_vmat = fvmat.read()
            # split by 2 empty lines i.e. maxclusters
            temp_vmat = pattern2.split(temp_vmat)

            while("" in temp_vmat):
                temp_vmat.remove("")  # remove empty blocks

            for clus_idx, mat in enumerate(temp_vmat):
                mat = mat.split('\n')  # split by 1 empty line i.e. subclusters
                mat_float = np.empty(list(map(int, mat[0].split(' '))))
                for idx, row in enumerate(mat[1:]):  # ignore first line
                    mat_float[idx] = list(map(float, row.split(' ')[:-1]))

                vmat[clus_idx] = mat_float

        return vmat

    @classmethod
    def read_eci(cls, eci_fname):

        # Read eci
        eci = {}

        with open(eci_fname, 'r') as feci:
            _ = next(feci)  # Ignore first line
            temp_eci = feci.read()
            temp_eci = temp_eci.split('\n')  # split by line

        for idx, eci_val in enumerate(temp_eci):
            if eci_val == '':
                continue
            eci[idx] = float(eci_val)

        return eci

    @classmethod
    def read_configs(cls, config_fname):

        pattern1 = re.compile("\n\n\n")
        pattern2 = re.compile("\n\n")
        # Read config.out
        configs = {}

        fconfig = open(config_fname, 'r')
        _ = next(fconfig)  # Ignore first line

        temp_config = fconfig.read()  # .split('\n\n')
        # split lines separated by 2 empty lines
        temp_config = pattern1.split(temp_config)

        for idx, config in enumerate(temp_config):
            if config == '':  # Check for spurious empty blocks
                continue
            num_points = int(config.split('\n')[0])  # number of subclusters
            inter = []
            # now split individual subclusters separated by 1 blank line
            config = pattern2.split(config)
            for i in range(num_points):
                line = config[i].split('\n')
                if i == 0:
                    length = int(line[1])
                else:
                    length = int(line[0])
                tmp_inter = np.array(
                    [(list(map(int, l.split(' ')[-2:]))) for l in line[-length:]])
                inter.append(tmp_inter)
            configs[idx] = {'inter': inter,
                            'num_of_subclus': num_points}

        return configs
