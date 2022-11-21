import itertools
import numpy as np
import ase.io
from energyfunctions import F_efficient
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

def no_get_random_structure(structure,lat_file,clusters_file):

    #convert str.in to POSCAR
    with open(f'{structure}/str.in','r') as f_in:
        with open(f'{structure}/str.in.poscar','w') as f_out:
            _ = subprocess.run(['str2poscar'],
                               stdin=f_in,
                               stdout=f_out
                              )
    str_in_atoms = ase.io.read(f'{structure}/str.in.poscar')

    new_atoms = str_in_atoms.get_atomic_numbers()
    np.random.shuffle(new_atoms)
    str_in_atoms.set_atomic_numbers(new_atoms)

    ase.io.write(f'{structure}/random.poscar',
                 str_in_atoms,
                 format='vasp',
                 sort=True,
                 wrap=True)

    with open(f'{structure}/random.poscar','r') as f_in:
        with open(f'{structure}/str.random','w') as f_out:
            _ = subprocess.run(['str2poscar', 'r'],
                               stdin=f_in,
                               stdout=f_out
                              )
    corr_random = subprocess.run(['corrdump', '-c', f'-cf={structure}/{clusters_file}', f'-s={structure}/str.random', f'-l={structure}/{lat_file}'],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 check=True
                                )
    corr_random = corr_random.stdout.decode('utf-8').split('\t')[:-1]
    corr_random = np.array(corr_random, dtype=np.float32)  # convert to arrays

    print(corr_random)
    return corr_random

def get_initial_trial(cluster_data,
                      corr_rnd,
                      T,
                      ord2disord_dist,
                      constraint,
                      trial_variance,
                      num_trials=100_000_00,
                      seed=42
                      ):

    rng = np.random.default_rng(seed)
    mult_arr = np.array(list(cluster_data.clustermult.values()))
    eci_arr = np.array(list(cluster_data.eci.values()))

    mults_eci = mult_arr * eci_arr

    all_vmat = np.vstack([vmat for vmat in cluster_data.vmat.values()])
    mults_config = np.array(
        list(itertools.chain.from_iterable(list(cluster_data.configmult.values()))))
    all_kb = np.array(list(itertools.chain.from_iterable([[kb for _ in range(
        len(cluster_data.configmult[idx]))] for idx, kb in cluster_data.kb.items()])))

    multconfig_kb = mults_config * all_kb
    assert all_vmat.shape == (len(multconfig_kb), len(mults_eci))
    vrhologrho = np.vectorize(lambda rho: rho * np.log(np.abs(rho)))

#    corr_trial = corr_rnd
#    fmin = F_efficient(corr_trial, mults_eci, multconfig_kb,
#                       all_vmat, vrhologrho, T)
#    corr_min = corr_rnd

    fixed_corrs = np.ones((num_trials,
                           len(cluster_data.single_point_clusters)+1
                           ))
    mean = np.zeros(cluster_data.num_clusters -
                    len(cluster_data.single_point_clusters) - 1)
    cov = trial_variance * \
        np.eye(cluster_data.num_clusters -
               len(cluster_data.single_point_clusters) - 1)
    random_corrs = np.random.multivariate_normal(mean, cov, num_trials)

    trial_corrs = np.hstack((fixed_corrs, random_corrs))
    trial_corrs = corr_rnd + trial_corrs
    trial_corrs = np.vstack((corr_rnd,trial_corrs))

    if constraint:
        def corr_validity(x): return (cluster_data.check_result_validity(x)) & (
            ord2disord_dist/2 - np.linalg.norm(x-corr_rnd) >= 0)
    else:
        def corr_validity(x): return (cluster_data.check_result_validity(x))

    valid_corrs = trial_corrs[np.apply_along_axis(corr_validity,
                                                  1,
                                                  trial_corrs
                                                  )
                              ]

    random_energy_eval = np.apply_along_axis(F_efficient,
                                             1,
                                             valid_corrs,
                                             mults_eci,
                                             multconfig_kb,
                                             all_vmat,
                                             vrhologrho,
                                             T
                                             )
    fmin, corr_min = np.amin(
        random_energy_eval), valid_corrs[np.argmin(random_energy_eval)]

    return fmin, corr_min
