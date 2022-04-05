import numpy as np
from scipy.optimize import curve_fit
import sys
import json

def sro_fit(func, results, coeff_in, method, verbose):
  try:
    with open(results, 'r') as fhandle:
      data = json.load(fhandle)
  except FileNotFoundError as e:
    sys.exit(f"Data file {results.split('/')[-1]} not found...")

  xdata = np.array([item.get('temperature') for item in data])
  ydata = np.array([item.get('F_cvm') - item.get('F_rnd') for item in data])

  try:
    p0 = np.loadtxt(coeff_in)
  except OSError:
    print('File containing initial coefficients not found..taking defaults...')
    p0 = None

  popt, pcov = curve_fit(func, xdata, ydata, method=method,)
  return popt, pcov



