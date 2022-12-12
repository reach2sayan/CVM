import re
import numpy as np
import random

def get_random_structure(structure):

    while True:
        with open(f'{structure}/str.in','r') as init_structure:
            lines = init_structure.readlines()
            positions = [[*line.strip().split(' ')] for lnum, line in enumerate(lines) if lnum > 5]
        atoms = [l[-1] for l in positions]
        random.shuffle(atoms)
        with open(f'{structure}/randstr.in','w') as random_structure:
            with open(f'{structure}/str.in','r') as init_structure:
                lines = init_structure.readlines()
                for lnum, line in enumerate(lines):
                    if lnum < 6:
                        random_structure.write(line)
                for position, atom in zip(positions,atoms):
                    random_structure.write(f'{" ".join(position[:-1])} {atom}\n')

        print('random structure generated..')
        yield

def read_clusters(clusters_fname) -> dict:

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

def read_kbcoeffs(kb_fname) -> dict:

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

def read_configmult(configmult_fname) -> dict:

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

def read_clustermult(clustermult_fname) -> dict:

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

def read_vmatrix(vmat_fname) -> dict:

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

def read_eci(eci_fname) -> dict:

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
    except FileNotFoundError:
        print('WARNING .....')
        print(
            f"No pre-existing {eci_fname.split('/')[-1]} file found")

    return eci

def read_configs(config_fname) -> dict:

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
