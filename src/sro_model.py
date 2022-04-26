"""
This is the function that can be modified by the user to define the function maps SRO correction aith respect to T
The first parameter should always be the independent variable (in this case T)
Subsequent parameters would be optimised for,

The output of the parameters will be the in the order they show up in the function defition
"""
import numpy as np

def sro_model(T, a0, a1, a2, C):
    return C*(np.exp(-a0*(T**(-1))) - 1)*(a1 + a2*(T**(-1)))

#def sro_model(T, a0, a1, a2, a3, C):
#    return C*np.abs(a0 + a1*(T**-1) + a2*(T**-2) + a3*(T**-3))

#inv_kB = 1/1.38064910-23
#
#def sro_model(T, a1, a2, a3, b1, b2, b3, C):
#    return -C*np.abs(1 + a1*np.exp(-b1*inv_kB*(T**-1)) + a2*np.exp(-b2*inv_kB*(T**-1)) + a3*np.exp(-b3*inv_kB*(T**-1)))

