import re
import numpy as np

def read_clusters(clusters_fname):
# Read clusters.out
    clusters = {}

    with open(clusters_fname,'r') as fclusters:
        temp_clusters = fclusters.read().split('\n\n') #Read blocks separated by 1 empty line

    for idx, cluster in enumerate(temp_clusters):
        if cluster == '': #Check for spurious empty blocks
            continue 
        line = cluster.split('\n') #If not empty split by lines
        multiplicity = int(line[0]) #1st line
        length = float(line[1]) #largest distance between two atoms
        num_points = int(line[2]) #type of cluster
        clusters[idx] = {'mult':multiplicity, 
                         'length':length, 
                         'type':num_points
                        }

    return clusters

def read_kbcoeffs(kb_fname):
# Read kb.out

    kb = {}
    fkb = open(kb_fname,'r')
    _ = next(fkb) #ignore first line

    temp_kb = fkb.read()
    temp_kb = temp_kb.split('\n') #split file linewise

    for idx, kbcoeff in enumerate(temp_kb):
        if kbcoeff == '': #check for spurious empty blocks
            continue
        kb[idx] = float(kbcoeff)

    fkb.close()
    return kb

def read_configcoef(configcoef_fname):
    # Read configcoeff.out
    configcoef = {}
    pattern1 = re.compile("\n\n\n")
    pattern2 = re.compile("\n\n")

    with open(configcoef_fname,'r') as fsubmult:
        _ = next(fsubmult) #ignore first line
        temp_submult = fsubmult.read()
        temp_submult = pattern2.split(temp_submult) #split lines into blocks separated by 2 empty lines

    for idx, submult in enumerate(temp_submult):
        submult = submult.split('\n') #split into number of subclusters
        while("" in submult) :
            submult.remove("") #remove empty blocks
        configcoef[idx] = list(map(float,submult[1:])) #also ignore 1st line of each block

    return configcoef

def read_vmatrix(vmat_fname):

    pattern1 = re.compile("\n\n\n")
    pattern2 = re.compile("\n\n")

    vmat = {}
    with open(vmat_fname,'r') as fvmat:
        _ = next(fvmat) #ignore first lie
        temp_vmat = fvmat.read()
        temp_vmat = pattern2.split(temp_vmat) #split by 2 empty lines i.e. maxclusters

        while("" in temp_vmat):
            temp_vmat.remove("") #remove empty blocks

        for clus_idx, mat in enumerate(temp_vmat):
            mat = mat.split('\n') #split by 1 empty line i.e. subclusters
            mat_float = np.empty(list(map(int, mat[0].split(' '))))
            for idx, row in enumerate(mat[1:]): #ignore first line
                mat_float[idx] = list(map(float,row.split(' ')[:-1]))

            vmat[clus_idx] = mat_float

    return vmat

def read_eci(eci_fname):

    # Read eci
    eci = {}

    with open(eci_fname,'r') as feci:
        _ = next(feci) #Ignore first line
        temp_eci = feci.read()
        temp_eci = temp_eci.split('\n') #split by line

    for idx, eci_val in enumerate(temp_eci):
        if eci_val == '':
            continue
        eci[idx] = float(eci_val)

    return eci

def read_configs(config_fname):

    pattern1 = re.compile("\n\n\n")
    pattern2 = re.compile("\n\n")
    # Read config.out
    configs = {}

    fconfig = open('config.out','r')
    _ = next(fconfig) #Ignore first line

    temp_config = fconfig.read()#.split('\n\n')
    temp_config = pattern1.split(temp_config) #split lines separated by 2 empty lines

    for idx, config in enumerate(temp_config):
        if config == '': #Check for spurious empty blocks
            continue
        num_points = int(config[0]) #number of subclusters
        config = pattern2.split(config[2:]) #now split individual subclusters separated by 1 blank line
        min_coords = []
        for _ in range(num_points):
            min_coords.append(config[_].split('\n')[0])
        configs[idx] = {'subclus': list(map(int,min_coords)),
                        'num_of_subclus': len(min_coords)}

    return configs
