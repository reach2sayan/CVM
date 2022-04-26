import subprocess
import numpy as np
from scipy.linalg import lstsq, LinAlgError
import argparse
import os
from pathlib import Path
import sys
from clusterdata import ClusterInfo

try:
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    print("scikit learn not found. Cannot use RANSACRegressor")

def fit_eci_scipy(clusters_fit, correlations, energies, structure):
    """
    Module to fit ECIs to a set of energies and corresponding correlations:
    Input: clusters_fit - cluster description for the fitting.
           correlations - dictionary containing all the correlations, index by the folder name
           energies     - corresponding energies also indexed by folder name
    Output: Fitted ECIs.
            Note, this gets padded with zeros to match the number of clusters of the full cluster description.
    """

    corrs, energies_array = map(np.array, zip(
        *[(correlations[key], energies[key]) for key in energies.keys()]))
    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
    mults_corrs = corrs*mults
    np.savetxt(f'{structure}/ref_correlations.in', mults_corrs)
    np.savetxt(f'{structure}/ref_energies.in', energies_array)
    try:
        eci_fit, _, _, _ = lstsq(
            mults_corrs, energies_array, lapack_driver='gelsd')
    except LinAlgError as ae:
        print(f'Issues with fitting ECI: {ae}')

    return eci_fit


def fit_eci_lsfit(clusters_fit, correlations, energies, structure):
    """
    Module to fit ECIs to a set of energies and corresponding correlations:
    Input:
        clusters_fit - cluster description for the fitting.
        correlations - dictionary containing all the correlations, index by the folder name
        energies     - corresponding energies also indexed by folder name
    Output:
        Fitted ECIs.
        Note, this gets padded with zeros to match the number of clusters of the full cluster description but not here
    """

    corrs, energies_array = map(np.array, zip(
        *[(correlations[key], energies[key]) for key in energies.keys()]))
    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
    mults_corrs = corrs*mults
    np.savetxt(f'{structure}/ref_correlations.in', mults_corrs)
    np.savetxt(f'{structure}/ref_energies.in', energies_array)

    eci_fit = subprocess.run(['lsfit', f'-x={structure}/ref_correlations.in', f'-y={structure}/ref_energies.in', '-colin'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             check=True
                             )
    eci_fit = np.fromstring(eci_fit.stdout.decode('utf-8'), sep='\n')

    return eci_fit

def fit_eci_ransac(clusters_fit, correlations, energies, structure):

    corrs, energies_array = map(np.array, zip(
        *[(correlations[key], energies[key]) for key in energies.keys()]))
    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
    mults_corrs = corrs*mults
    np.savetxt(f'{structure}/ref_correlations.in', mults_corrs)
    np.savetxt(f'{structure}/ref_energies.in', energies_array)

    lr = LinearRegression(fit_intercept = False)
    ransac = RANSACRegressor(base_estimator=lr,loss='squared_loss')
    ransac.fit(mults_corrs, energies_array)
    for idx, outlier in enumerate(np.logical_not(ransac.inlier_mask_)):
        if outlier:
            print(f'Outlier correlation: {mults_corrs[idx]}')
            print(f'Outlier Energy: {energies_array[idx]}')
    print(f'Mean Square Error: {mean_squared_error(energies_array[ransac.inlier_mask_], mults_corrs[ransac.inlier_mask_] @ ransac.estimator_.coef_ )}')
    print(f'R2 score: {r2_score(energies_array[ransac.inlier_mask_], mults_corrs[ransac.inlier_mask_] @ ransac.estimator_.coef_ )}')
    return ransac.estimator_.coef_

if __name__ == '__main__':

    structure = os.getcwd()
    path = Path(structure)
    phase = str(path.parent.absolute())

    parser = argparse.ArgumentParser(description='CVM SRO Error Correction Code by Sayan Samanta and Axel van de Walle',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    )

    clus_fit_params = parser.add_argument_group(
        "Parameters related to cluster fitting")
    clus_fit_params.add_argument('--clusters',
                                 default='clusters.out',
                                 help="contain the cluster description for fitting the ECIs ",
                                )
    clus_fit_params.add_argument('--eci',
                                 default='eci.out',
                                 help="file to output fitted ECIs ",
                                )
    clus_fit_params.add_argument('--use_lsfit', '-lsfit',
                                 action='store_true',
                                 default=False,
                                 help="Use ATAT lsfit to fit ECI",
                                )
    clus_fit_params.add_argument('--use_ransac', '-ransac',
                                 action='store_true',
                                 default=True,
                                 help="Use sklearn RANSACRegressor ECI",
                                )
    clus_fit_params.add_argument('--reference_structures', '-ref',
                                 default='references.in',
                                 help="contains the names of the reference structures used to fit ECI",
                                )
    args = parser.parse_args()

    print(
        'Fitting ECIs.\n')
    energies = {}
    correlations = {}
    try:
        with open(f'{structure}/{args.reference_structures}', 'r') as fref:
            reference_structures = [
                structure.rstrip() for structure in fref.readlines()]
    except FileNotFoundError as fnotf:
        print(fnotf)
        print(f"{args.reference_structures} not found. Exiting.\n")
        sys.exit(1)

    clusters = ClusterInfo(args.clusters, None, None, None, None, None,)


    for struc in reference_structures:
        try:
            with open(f'{struc}/energy', 'r') as energy_file:
                e = float(energy_file.readline())

            num_atoms = sum(1 for line in open(f"{struc}/str.in")) - 6
            energies[struc] = e/num_atoms
            with open(f'{struc}/energy_atom', 'w') as feperatom:
                feperatom.write(str(e/num_atoms))
                corr = subprocess.run(['corrdump', '-c', f'-cf={structure}/{args.clusters_fit}', f'-s={struc}/str.in', f'-l={structure}/{args.lat}'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      check=True
                                     )
                # convert from bytes to string list
                corr = corr.stdout.decode('utf-8').split('\t')[:-1]
                corr = np.array(corr, dtype=np.float32)  # convert to arrays
                assert corr[0] == 1
                correlations[struc] = corr

            print(f"Found structure {struc.split('/')[-1]} with energy : {e/num_atoms}")
            print(f"Correlations {struc.split('/')[-1]} : {corr}\n")

        except FileNotFoundError as fnotf:
            print(fnotf)
            print(
                f"Energy of reference structure {struc.split('/')[-1]} not found. Probably some error in DFT calc")

    if args.use_lsfit:
        ecis = fit_eci_lsfit(
            clusters, correlations, energies, structure)
    elif args.use_ransac:
        ecis = fit_eci_ransac(clusters,
                                 correlations,
                                 energies,
                                 structure
                                )

    print(f'Fitted ECIs : {ecis}')
    print(f'Writing Fitted ECIs to file {args.eci}')
    with open(f'{args.eci}', 'w') as eci_file:
        eci_file.write(f'{clusters.num_clusters}\n')
        for eci in ecis:
            eci_file.write(f'{eci}\n')

    print('\n========================================\n')
