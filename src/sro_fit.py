import numpy as np
from scipy.optimize import curve_fit
import sys
import json
import inspect
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
)
from sympy import *

eV2J = 96491.5666370759
def sro_fit(func, results, coeff_in, inJoules=False,num_atoms=1):
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

    xdata = np.array([item.get('temperature') for item in data if item.get('temperature') != 0])
    ydata = np.array([item.get('F_cvm') - item.get('F_rnd') for item in data if item.get('temperature') != 0])*num_atoms
    if inJoules:
        ydata = ydata*eV2J
    try:
        p0 = np.loadtxt(coeff_in)
    except OSError:
        print('File containing initial coefficients not found..taking defaults...')
        p0 = None

    popt, pcov = curve_fit(f=func, xdata=xdata, ydata=ydata, p0=p0,method='dogbox', maxfev=10000000)

    for line in inspect.getsource(func).split('\n'):
        if 'return' in line:
            return_line = line.replace('return','').strip()
    return_line_replaced = str(return_line)
    for index, varname in enumerate(func.__code__.co_varnames[1:]):
        return_line_replaced = return_line_replaced.replace(varname,f'{popt[index]}')

    func_replace = {'np.exp':'exp',
                    'np.abs':'ABS',
                    'np.sin':'SIN'}
    for old_item, new_item in func_replace.items():
        return_line_replaced = return_line_replaced.replace(old_item,new_item)

    xcont = np.linspace(min(xdata), max(xdata), 1000)[:, np.newaxis]
    ycont = func(xcont,*popt)
    return_line_replaced = str(parse_expr(return_line_replaced,
                                      transformations=standard_transformations+(implicit_multiplication,)).expand(basic=True))
    return_line_replaced = return_line_replaced.replace('/T','*(T**(-1))')
    return_line_replaced = return_line_replaced.replace(' ','')
    return_line_replaced = return_line_replaced.replace('++','+')
    return_line_replaced = return_line_replaced.replace('+-','-')
    return_line_replaced = return_line_replaced.replace('-+','-')
    return_line_replaced = return_line_replaced.replace('--','+')
    return_line_replaced = return_line_replaced.replace('exp','EXP')

    print(f'Output to func:\n{return_line_replaced}')
    np.savetxt('sro_fitting_calculated.out',np.hstack((xcont, ycont)))
    np.savetxt('sro_fitting_data.out',np.hstack((xdata[:,np.newaxis], ydata[:,np.newaxis])))

    return popt, pcov, return_line, return_line_replaced, func.__code__.co_varnames[1:]
