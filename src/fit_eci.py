import subprocess
import numpy as np
from scipy.linalg import lstsq, LinAlgError


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
    try:
        eci_fit, _, _, _ = lstsq(
            mults_corrs, energies_array, lapack_driver='gelsd')
    except LinAlgError as ae:
        print(f'Issues with fitting ECI: {ae}')

    return eci_fit


def fit_eci_lsfit(clusters_fit, correlations, energies, structure):
    """
    Module to fit ECIs to a set of energies and corresponding correlations:
        Input: clusters_fit - cluster description for the fitting.
        correlations - dictionary containing all the correlations, index by the folder name
        energies     - corresponding energies also indexed by folder name
        Output: Fitted ECIs. Also provides a plot to check the result of the fit.
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

    fitted_energies = mults_corrs @ eci_fit

    return eci_fit


# plt.style.use('seaborn-paper')
#plt.rc('font', family='serif')
#plt.rc('xtick', labelsize='x-small')
#plt.rc('ytick', labelsize='x-small')
#plt.rc('text', usetex=True)
#fitted_energies = mults_corrs @ eci_fit
#    plt.plot(energies_array, energies_array, 'X',
#             label='Ab-initio Energies')
#    plt.plot(energies_array, fitted_energies,
#             'd', label='Fitted Energies')
#    plt.xlabel('Energies Calculated (in eV)')
#    plt.ylabel('Energies Fitted (in eV)')
#    plt.title('ECI Fit Results')
#    plt.legend()
#    print(f'Energies FP: {energies_array}\nEnergies Fitted: {fitted_energies}')
#    plt.tight_layout()
#    plt.savefig('eci_fit_results_scipy.svg', dpi=300)
