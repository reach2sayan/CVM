import numpy as np
import re

def read_clusters(clusters_fname):
    clusters = {}

    with open(clusters_fname,'r') as fclusters:
        temp_clusters = fclusters.read().split('\n\n')

    for idx, cluster in enumerate(temp_clusters):
        if cluster == '':
            continue
        line = cluster.split('\n')
        multiplicity = int(line[0])
        length = float(line[1])
        num_points = int(line[2])
        clusters[idx] = {'mult':multiplicity, 'length':length, 'type':num_points}

    return clusters, len(clusters)

def read_configs(config_fname):
    configs = {}

    pattern1 = re.compile("\n\n\n")
    pattern2 = re.compile("\n\n")

    with open('config.out','r') as fconfig:
        _ = next(fconfig)
        temp_config = fconfig.read()

    temp_config = pattern1.split(temp_config)

    for idx, config in enumerate(temp_config):
        if config == '':
            continue
        num_points = int(config[0])
        config = pattern2.split(config[2:])
        min_coords = []
        for _ in range(num_points):
            min_coords.append(config[_].split('\n')[0])
        configs[idx] = {'subclus': list(map(int,min_coords)), 'num_of_subclus': len(min_coords)}

    return configs

def read_kb_coefficients(kb_fname):
    kbcoeff = {}

    with open(kb_fname,'r') as fkb:
        _ = next(fkb)

        temp_kb = fkb.read()
        temp_kb = temp_kb.split('\n')

    for idx, kb in enumerate(temp_kb):
        if kb == '':
            continue
        kbcoeff[idx] = float(kb)

    return kbcoeff


def read_vmat(vmat_fname):
    vmat = {}

    with open(vmat_fname) as fvmat:
        _ = next(fvmat)

        temp_vmat = fvmat.read()
        temp_vmat = pattern2.split(temp_vmat)

    while("" in temp_vmat) :
        temp_vmat.remove("")

    for clus_idx, mat in enumerate(temp_vmat):
        mat = mat.split('\n')
        mat_float = np.empty(list(map(int, mat[0].split(' '))))
        
        for idx, row in enumerate(mat[1:]):
            mat_float[idx] = list(map(float,row.split(' ')[:-1]))

        vmat[clus_idx] = mat_float

    return vmat

def read_eci(eci_fname):


