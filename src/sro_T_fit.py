#!/usr/bin/env python3

import sys
import json
from lmfit import Model, Parameters
import numpy as np

def read_results(fname):

    try:
        with open(fname, 'r') as fhandle:
            data = json.load(fhandle)
    except Exception as e:
        print(f'Error opening data file {fname}...')
        exit(1)

    xdata = np.array([item.get('temperature') for item in data])
    ydata = np.array([item.get('F_cvm') - item.get('F_rnd') for item in data])

    return xdata, ydata

def read_coeffs(fname, degree):

    trial = []
    try:
        with open(fname, 'r') as handle:
            data = handle.readlines()
        for idx, d in enumerate(range(degree+1)):
            tmp = data[idx].strip().split(',')
            trial.append([float(tmp[0]), float(tmp[-1])])
        trial.append([0.01, 0])
        return np.array(trial)
    except FileNotFoundError as e:
        print(f"File {fname.split('/')[-1]} not found. Initialising with defaults")
        for d in range(degree+1):
            if d == 0:
                trial.append([1, 0])
            else:
                trial.append([1*(-1)**d, 0.01])
        trial.append([0.01, 0])
        return np.array(trial)

def sro_T_fit(func, degree, results_file, coeff_in, skip_expo, method, verbose):

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
        sys.exit('WARNING. The fitted function does not go to zero as T tends to infinity...')

    print('Fitting Results:')
    print(results.fit_report())

    return results.best_values


    # plt.style.use('seaborn-paper')
    #fig = plt.figure(dpi=100, figsize=(5, 4),)

    #xr = np.linspace(0,int(max(xdata)),100)
    #yr = model.eval(T=xr,**best_params)
    #plt.plot(xdata, ydata, '.', label='data')
    # plt.plot(xr,yr,label='fit')
    # plt.xlabel('T')
    #plt.ylabel(r'F$_{cvm}$ - F$_{disordered}$')
    # plt.grid(True)
    # plt.legend()

    # plt.savefig(f'{args.out}.svg')
