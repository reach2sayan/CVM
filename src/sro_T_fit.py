"""
Module containing code to fit ECI
"""
import sys
import json
try:
    from lmfit import Model, Parameters
except ImportError as ie:
    print('WARNING: lmfit not found. SRO T fit cannot be obtained.')
import numpy as np


def read_results(fname):
    """
    Parses the json file to extract relevanle arrays to fit
    """
    try:
        with open(fname, 'r') as fhandle:
            data = json.load(fhandle)
    except FileNotFoundError as e:
        sys.exit(f"Data file {fname.split('/')[-1]} not found...")

    xdata = np.array([item.get('temperature') for item in data])
    ydata = np.array([item.get('F_cvm') - item.get('F_rnd') for item in data])

    return xdata, ydata


def read_coeffs(fname, degree):
    """
    Reads the initial coefficient from file.
    Creates default is file not found
    """

    trial = []
    try:
        with open(fname, 'r') as handle:
            data = handle.readlines()
        for idx, _ in enumerate(range(degree+1)):
            tmp = data[idx].strip().split(',')
            trial.append([float(tmp[0]), float(tmp[-1])])
        trial.append([0.01, 0])
        return np.array(trial)
    except FileNotFoundError:
        print(
            f"File {fname.split('/')[-1]} not found. Initialising with defaults")
        for d in range(degree+1):
            if d == 0:
                trial.append([1, 0])
            else:
                trial.append([1*(-1)**d, 0.01])
        trial.append([0.01, 0])
        return np.array(trial)


def sro_T_fit(func, degree, results_file, coeff_in, skip_expo, method, verbose):
    """
    Function to perform the fit
    Input:
        func - function to fit to
        degree - max degree of the polynomial
        results_file - file to read optimised data from
        coeff_in - Initial guess for the parameters
        skip_expo - Flag to have a constant exponential terms across all powers
        method - default Least-Squares minimization, using Trust Region Reflective method
        verbose- verbosity
    Output:
        Best fit parameter values
    """
    xdata, ydata = read_results(results_file)
    model = Model(func)
    params = Parameters()
    ini_coeff = read_coeffs(coeff_in, degree)
    for i in range(degree+1):

        if i == 0:
            params.add(f'coeff_{i}', value=ini_coeff[0][0], vary=False,)
            params.add(f'exp_{i}', value=ini_coeff[0][-1], min=0, vary=False)
        elif i == degree:
            params.add(f'coeff_{i}', expr='-'+'-'.join(f'coeff_{i}' for i in range(degree)),
                       value=1*(-1)**i,
                       min=-5, max=5,
                       )
            if skip_expo:
                params.add(
                    f'exp_{i}', value=ini_coeff[i][-1], min=0, vary=False)
            else:
                params.add(
                    f'exp_{i}', value=ini_coeff[i][-1], min=0, vary=True)
        else:
            params.add(
                f'coeff_{i}', value=ini_coeff[i][0], min=-5, max=5, vary=True)
            if skip_expo:
                params.add(
                    f'exp_{i}', value=ini_coeff[i][-1], min=0, vary=False)
            else:
                params.add(
                    f'exp_{i}', value=ini_coeff[i][-1], min=0, vary=True)

    params.add('C', value=ini_coeff[-1][0])
    params.add('degree', value=degree, vary=False)

    print('Preparing Parameters for fit...')
    print(params.pretty_print())

    print('Performing fit...')

    try:
        results = model.fit(ydata,
                            params=params,
                            T=xdata,
                            method=method,
                            verbose=verbose
                            )
    except Exception as e:
        sys.exit('Error Fitting...\n', e)
    try:
        assert np.isclose(model.eval(T=np.inf, **results.best_values), 0.0)
    except AssertionError:
        sys.exit(
            'WARNING. The fitted function does not go to zero as T tends to infinity...')

    print('Fitting Results:')
    print(results.fit_report())
    xr = np.linspace(0, int(max(xdata)), 100)
    yr = model.eval(T=xr, **results.best_values)
    np.savetxt('x_sro.dat', xdata)
    np.savetxt('y_sro.dat', ydata)
    np.savetxt('x_sro_fit.dat', xr)
    np.savetxt('y_sro_fit.dat', yr)

    return results.best_values
