#!/usr/bin/python3

from subprocess import Popen, PIPE
from pathlib import Path
import os
import numpy as np
import ase.io
from clusterdata import ClusterInfo

rng = np.random.default_rng()


def create_structure(label):

    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, str(label)), exist_ok=True)

    for structure in ['str.in', 'str.out']:

        # convert str.out and str.in to POSCAR
        with open(f'{cwd}/{structure}', 'r') as frelax:
            relax_str = frelax.read()

        p = Popen(['str2poscar'], stdout=PIPE, stdin=PIPE)
        p.stdin.write(str.encode(relax_str))
        poscar = p.communicate()[0].decode('utf-8')

        with open(f'{cwd}/{structure}.POSCAR', 'w') as fposcar:
            fposcar.write(poscar)

        # read tmp.POSCAR to ase atoms for creating other structures
        atoms = ase.io.read(f'{cwd}/{structure}.POSCAR')

        elems = atoms.get_chemical_symbols()
        rng.shuffle(elems)
        atoms.set_chemical_symbols(elems)

        ase.io.write(
            f'{cwd}/{label}/{structure}_shuffled.POSCAR', atoms, sort=True)

        # convert to str.out
        p = Popen(['str2poscar', '-r'], stdout=PIPE, stdin=PIPE)
        with open(f'{cwd}/{label}/{structure}_shuffled.POSCAR', 'r') as fout:
            out_str = fout.read()

        p.stdin.write(str.encode(out_str))
        out = p.communicate()[0].decode('utf-8')
        with open(f'{cwd}/{label}/{structure}', 'w') as fout:
            fout.write(out)

        with open(f'{cwd}/{label}/wait', 'w') as fwait:
            pass

    print(f'Created Structure {label}.')
    return


if __name__ == '__main__':

    cwd = os.getcwd()
    path = Path(cwd)
    phase = path.parent.absolute()

    clfit = ClusterInfo(clusters_fname=f'{phase}/clusters_fit.out',
                        eci_fname=f'{phase}/eci.out',
                        cluster_only=True
                        )
    print(
        f'Found {clfit.num_clusters} no. of clusters to fit. Creating as many structures...')
    ref_list = open('references.in', 'w')
    ref_list.write(f'{cwd}')
    for label in range(clfit.num_clusters*2):
        create_structure(label)
        ref_list.write(f'{cwd}/{label}\n')
    ref_list.close()
