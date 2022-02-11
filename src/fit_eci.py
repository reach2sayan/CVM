#!/usr/bin/python3
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)


def fit_eci(clusters_fit, correlations, energies):
    """
    Module to fit ECIs to a set of energies and corresponding correlations:
    Input: clusters_fit - cluster description for the fitting.
           correlations - dictionary containing all the correlations, index by the folder name
           energies     - corresponding energies also indexed by folder name
    Output: Fitted ECIs. Also provides a plot to check the result of the fit.
            Note, this gets padded with zeros to match the number of clusters of the full cluster description but not here
    """

    def func(ecis, mult_corrs, energy):
        return (mult_corrs @ ecis) - energy

    corrs, energies_array = map(np.array, zip(
        *[(correlations[key], energies[key]) for key in energies.keys()]))
    ecis = np.array(list([clusters_fit.eci[index] for index in clusters_fit.clusters.keys(
    ) if clusters_fit.clusters[index]['type'] <= 2]))
    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
    mults_corrs = corrs*mults
    eci_fit = least_squares(func,
                            ecis,
                            method='trf',
                            x_scale='jac',
                            jac='3-point',
                            args=(mults_corrs, energies_array),
                            verbose=2,
                            )
    try:
        assert eci_fit.success
    except AssertionError as ae:
        print(f'Issues with fitting ECI: {ae}')

    fitted_energies = mults_corrs @ eci_fit.x
    plt.plot(energies_array, energies_array, 'X',
             label='Ab-initio Energies')
    plt.plot(energies_array, fitted_energies,
             'd', label='Fitted Energies')
    plt.xlabel('Energies Calculated (in eV)')
    plt.ylabel('Energies Fitted (in eV)')
    plt.title('ECI Fit Results')
    plt.legend()
    print(f'Energies FP: {energies_array}\nEnergies Fitted: {fitted_energies}')
    plt.tight_layout()
    plt.savefig('eci_fit_results.svg', dpi=300)

    return eci_fit.x
