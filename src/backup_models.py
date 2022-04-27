
def sro_model(T, a0, a1, a2,):
    return (np.exp(-a0*(T**(-1))) - 1)*(a1 + a2*(T**(-1)))

def sro_model(T,a0,a1,a2,a3,b1,b2):
    """
    Model definition for the SRO correction to T function
    """
    return a0 + a1*np.exp(b1*(T**(-1))) + a2*(T**(-1))*np.exp(b2*(T**(-1))) + a3*(T**(-1))

def sro_model(T,a0,a1,b0):
    """
    Model definition for the SRO correction to T function
    """
    return a0 - a0*np.exp(b0*(T**(-1))) + a1*(T**(-1))*np.exp(b0*(T**(-1))) - a1*(T**(-1))

def sro_model(T, a0, a1, a2,):
       return (np.exp(-a0*(T**(-1))) - 1)*(a1 + a2*(T**(-1)))

def sro_model(T, a0, a1, a2, a3, b2, b3):
   return a0 + a1*(T**(-1)) + a2*np.exp(b2*(T**(-1))) + a3*(T**(-1))*np.exp(b3*(T**(-1)))

def sro_model(T, a0, a1, a2, a3, C):
       return C*np.abs(a0 + a1*(T**-1) + a2*(T**-2) + a3*(T**-3))

inv_kB = 1/1.38064910-23

def sro_model(T, a1, a2, a3, b1, b2, b3, C):
       return -C*np.abs(1 + a1*np.exp(-b1*inv_kB*(T**-1)) + a2*np.exp(-b2*inv_kB*(T**-1)) + a3*np.exp(-b3*inv_kB*(T**-1)))

