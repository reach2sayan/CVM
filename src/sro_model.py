"""
This is the function that can be modified by the user to define the function maps SRO correction aith respect to T
The first parameter should always be the independent variable (in this case T)
Subsequent parameters would be optimised for,

The output of the parameters will be the in the order they show up in the function defition
"""
import numpy as np

def sro_model(T, a0, a1, a2,):
    return (np.exp(-a0*(T**(-1))) - 1)*(a1 + a2*(T**(-1)))
