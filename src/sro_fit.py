import numpy as np
from scipy.optimize import curve_fit
import sys
import json

def sro_fit(func, results, coeff_in, method,):
    """
    Function to fit SRO correction as a function of T
    Inputs:
        func - the fitting function, the first parameter is the independent parameter
        results - json file containing the results of a CVM optimisation
        coeff_in - Optional parameter for initial guesses
        method - method of the fit
        verbose - verbosity
    Output:
        popt - Best fit params
        pcov - covariance of the parameters
    """
    try:
        with open(results, 'r') as fhandle:
            data = json.load(fhandle)
    except FileNotFoundError:
        sys.exit(f"Data file {results.split('/')[-1]} not found...")

    xdata = np.array([item.get('temperature') for item in data])
    ydata = np.array([item.get('F_cvm') - item.get('F_rnd') for item in data])

    try:
        p0 = np.loadtxt(coeff_in)
    except OSError:
        print('File containing initial coefficients not found..taking defaults...')
        p0 = None

    popt, pcov = curve_fit(f=func, xdata=xdata, ydata=ydata, method=method,)

    xcont = np.linspace(min(xdata), max(xdata), 1000)[:, np.newaxis]
    ycont = func(xcont,*popt)

    np.savetxt('sro_fitting_calculated.out',np.hstack((xcont, ycont)))
    np.savetxt('sro_fitting_data.out',np.hstack((xdata[:,np.newaxis], ydata[:,np.newaxis])))

    return popt, pcov
