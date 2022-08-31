import numpy as np
import re
import sys


class ClusterInfo:
    """
    Class containing the clusters information.
    Input: Filenames containing
           1. Cluster File,
           2. ECI
           3. Kikuchi-Baker Coefficients
           4. Cluster Coefficients
           4. Config Coefficients
           5. Config Information (this is not particularly used in calculations)
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
                 eci_fname=None,
                 kb_fname=None,
                 clustermult_fname=None,
                 configmult_fname=None,
                 config_fname=None,
                 vmat_fname=None,
                ):

        self.clusters = ClusterInfo.read_clusters(clusters_fname)
        self.kb = ClusterInfo.read_kbcoeffs(kb_fname)
        self.configmult = ClusterInfo.read_configmult(configmult_fname)
        self.clustermult = ClusterInfo.read_clustermult(
            clustermult_fname, len(self.clusters))
        self.configs = ClusterInfo.read_configs(config_fname)
        self.vmat = ClusterInfo.read_vmatrix(vmat_fname)
        self.eci = ClusterInfo.read_eci(eci_fname, len(self.clusters))

    def print_rho(self, corrs):

        for vmat in self.vmat.values():
            with np.printoptions(precision=3,suppress=True):
                print(np.array(vmat @ corrs))
        return

    def __repr__(self):

        print("Clusters:")
        print("{0:<19s}|{1:<19s}|{2:<19s}|{3:<19s}".format("Index","Type", "Multiplicity","Radius"))
        for idx, cluster in self.clusters.items():
            assert self.clustermult[idx] == cluster['mult']
            print("{0:<19d}|{1:<19d}|{2:<19d}|{3:<19.5f}".format(idx, cluster['type'], cluster['mult'], cluster['length']))

        print("\nConfigs:")
        print("{0:<19s}|{1:<19s}".format("Index","No. of subconfigs"))
        for idx, config in self.configs.items():
            assert len(self.configmult[idx]) == config['num_of_subclus']
            print("{0:<19d}|{1:<19d}".format(idx, config['num_of_subclus'],))

        return ''


    @property
    def num_configs(self):
        return len(self.configmult)

    @property
    def single_point_clusters(self):
        return [cluster_idx for cluster_idx, cluster in self.clusters.items() if cluster['type'] == 1]

    @property
    def num_clusters(self):
        return len(self.clusters)

    def check_result_validity(self, corrs):
        try:
          for vmat in self.vmat.values():
            rho = vmat @ corrs
            assert np.all([x >= 0.0 and x <= 1.0 for x in rho])
        except Exception:
          return False
        return True


    @classmethod
    def read_clusters(cls, clusters_fname):

        clusters = {}

        try:
            with open(clusters_fname, 'r') as fclusters:
                # Read blocks separated by 1 empty line
                temp_clusters = fclusters.read().split('\n\n')
        except FileNotFoundError as fnfe:
            print(
                f"WARNING: CLuster description file {clusters_fname.split('/')[-1]} not found. ")
            return None
        except TypeError:
            return None

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
        try:
            fkb = open(kb_fname, 'r')
            _ = next(fkb)  # ignore first line

            temp_kb = fkb.read()
            fkb.close()
        except FileNotFoundError as fnfe:
            print(
                f"WARNING: Kikuchi-Barker coefficients file {kb_fname.split('/')[-1]} not found. ")
            return None
        except TypeError:
            return None

        temp_kb = temp_kb.split('\n')  # split file linewise
        for idx, kbcoeff in enumerate(temp_kb):
            if kbcoeff == '':  # check for spurious empty blocks
                continue
            kb[idx] = float(kbcoeff)

        return kb

    @classmethod
    def read_configmult(cls, configmult_fname):

        configmult = {}
        pattern1 = re.compile("\n\n\n")
        pattern2 = re.compile("\n\n")

        try:
            with open(configmult_fname, 'r') as fsubmult:
                _ = next(fsubmult)  # ignore first line
                temp_submult = fsubmult.read()
                # split lines into blocks separated by 2 empty lines
                temp_submult = pattern2.split(temp_submult)
        except FileNotFoundError as fnfe:
            print(
                f"WARNING: Config Multiplicities file {configmult_fname.split('/')[-1]} not found. ")
            return None
        except TypeError:
            return None

        for idx, submult in enumerate(temp_submult[:-1]):
            submult = submult.split('\n')  # split into number of subclusters
            while("" in submult):
                submult.remove("")  # remove empty blocks
            # also ignore 1st line of each block
            configmult[idx] = list(map(float, submult[1:]))

        return configmult

    @classmethod
    def read_clustermult(cls, clustermult_fname, numclus):

        # Read cluster multiplicities
        clustermult = {}
        try:
            with open(clustermult_fname, 'r') as fcm:
                _ = next(fcm)  # Ignore first line
                temp_mult = fcm.read()
                temp_mult = temp_mult.split('\n')  # split by line

            for idx, mult in enumerate(temp_mult):
                if mult == '':
                    continue
                clustermult[idx] = float(mult)
        except FileNotFoundError:
            print(
                f"WARNING: Cluster Multiplicities file {clustermult_fname.split('/')[-1]} not found. ")
            return None
        except TypeError:
            return None

        return clustermult

    @classmethod
    def read_vmatrix(cls, vmat_fname):

        pattern1 = re.compile("\n\n\n")
        pattern2 = re.compile("\n\n")

        vmat = {}
        try:
            with open(vmat_fname, 'r') as fvmat:
                _ = next(fvmat)  # ignore first lie
                temp_vmat = fvmat.read()
        except FileNotFoundError as fnfe:
            print(
                f"WARNING: Vmat file {vmat_fname.split('/')[-1]} not found. ")
            return None
        except TypeError:
            return None

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
    def read_eci(cls, eci_fname, numclus):

        # Read eci
        eci = {}
        try:
            with open(eci_fname, 'r') as feci:
                _ = next(feci)  # Ignore first line
                temp_eci = feci.read()
                temp_eci = temp_eci.split('\n')  # split by line

            print(
                f"Reading ECIs from existing file {eci_fname.split('/')[-1]}.")
            for idx, eci_val in enumerate(temp_eci):
                if eci_val == '':
                    continue
                eci[idx] = float(eci_val)
        except (FileNotFoundError, TypeError):
            if eci_fname is not None:
                print('WARNINGGGGGGGG.....')
                print(
                    f"No pre-existing {eci_fname.split('/')[-1]} file found. Instantiating with all zeros")
            else:
                print('WARNINGGGGGGGG.....')
                print(
                    f"Instantiating ECI with all zeros")

            temp_eci = np.array([0]*numclus)

            for idx, eci_val in enumerate(temp_eci):
                eci[idx] = eci_val
            return eci

        return eci

    @classmethod
    def read_configs(cls, config_fname):

        pattern1 = re.compile("\n\n\n")
        pattern2 = re.compile("\n\n")
        # Read config.out
        configs = {}

        try:
            with open(config_fname, 'r') as fconfig:
                _ = next(fconfig)  # Ignore first line

                temp_config = fconfig.read()  # .split('\n\n')
        except FileNotFoundError as fnfe:
            print(
                f"WARNING: Config Description file {config_fname.split('/')[-1]} not found. Since this is not explicitly used in calculation. The programs shall continue.")
            return None
        except TypeError:
            return None

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
