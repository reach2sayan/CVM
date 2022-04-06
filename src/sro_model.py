"""
This is the function that can be modified by the user to define the function maps SRO correction aith respect to T
The first parameter should always be the independent variable (in this case T)
Subsequent parameters would be optimised for,

The output of the parameters will be the in the order they show up in the function defition
"""
import numpy as np

kB = 8.617330337217213e-05


def sro_model(T, a1, a2, a3, b1, b2, b3, C):
    return -C*np.abs(1 + a1*np.exp(-b1/(kB*T)) + a2*np.exp(-b2/(kB*T)) + a3*np.exp(-b3/(kB*T)))
