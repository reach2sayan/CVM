#!/usr/bin/python3
import os
import random

def create_structure(label):
    """
    Module to create a subdirectory named <label>
    The str.in and str.out files from the parent structure is taken
    and any two unlike atoms are swapped to create new structures
    that are close to the ordered structure.
    This is done to create datapoints to fit ECIs for a local clusters expansion
    on this structure
    """

    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, str(label)), exist_ok=True)

    for structure in ['str.in', 'str.out']:
        cell = []
        positions = []
        with open(f'{cwd}/{structure}', 'r') as f:
            for line in f.readlines():
                if len(line.rstrip().split(' ')) == 4:
                    positions.append(line.rstrip().split(' '))
                else:
                    cell.append(line.rstrip())

            while True:
                rnd_index1 = random.randint(0, len(positions)-1)
                rnd_index2 = random.randint(0, len(positions)-1)

                if positions[rnd_index1][3] != positions[rnd_index2][3]:
                    tmp = positions[rnd_index1][3]
                    positions[rnd_index1][3] = positions[rnd_index2][3]
                    positions[rnd_index2][3] = tmp
                    break

            with open(f'{cwd}/{label}/{structure}', 'w') as flabel:
                for line in cell:
                    flabel.write(line+'\n')
                for line in positions:
                    flabel.write(" ".join(line)+'\n')
            with open(f'{cwd}/{label}/wait','w'):
                pass
