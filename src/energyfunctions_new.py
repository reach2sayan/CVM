from typing import Callable
import numpy as np
import math
import sys

_kB = 8.617330337217213e-05
def F(corrs: np.ndarray,
      mults_eci: np.ndarray,
      multconfig_kb: np.ndarray,
      all_vmat: np.ndarray,
      vect_rhologrho: Callable[[np.ndarray], np.ndarray],
      temp: float
     ) -> float:
    """
    Input:
        corrs - Correlations
        vmat  - V-Matrix
        clusters - Maximal Cluster Information (multiplicity, longest neighbor length, no. of points)
        configs - Not used
        clustermult - Multiplicities of clusters
        configmult - Multiplicities of configurations
        T - Temperature
        eci - ECI's

    Output:
        F = H + kB*T*SUM(rho * log(rho))
        """

    H = mults_eci @ corrs
    S = multconfig_kb @ vect_rhologrho((all_vmat @ corrs) + sys.float_info.epsilon)

    return H + _kB*temp*S

def F_jacobian(corrs: np.ndarray,
               mults_eci: np.ndarray,
               multconfig_kb: np.ndarray,
               all_vmat: np.ndarray,
               vect_rhologrho: Callable[[np.ndarray], np.ndarray],
               temp: float
              ) -> np.ndarray:
    """
    Input: 
        corrs - Correlations
        vmat  - V-Matrix
        clusters - Maximal Cluster Information (multiplicity, longest neighbor length, no. of points)
        configs - Not used
        clustermult - Multiplicities of clusters
        configmult - Multiplicities of configurations
        T - Temperature
        eci - ECI's

    Output:
        Vector representation gradient of F with Corrs
        [dF/dcorr0, dF/dcorr1, ...]
    """

    dH = mults_eci
    dS = all_vmat.T @ (multconfig_kb * (1 + np.log(np.abs((all_vmat @ corrs) + sys.float_info.epsilon))))

    return dH + _kB*temp*dS

def F_hessian(corrs: np.ndarray,
               mults_eci: np.ndarray,
               multconfig_kb: np.ndarray,
               all_vmat: np.ndarray,
               vect_rhologrho: Callable[[np.ndarray], np.ndarray],
               temp: float
              ) -> np.ndarray:
    """
    Input:
        corrs - Correlations
        vmat  - V-Matrix
        clusters - Maximal Cluster Information (multiplicity, longest neighbor length, no. of points)
        configs - Not used
        clustermult - Multiplicities of clusters
        configmult - Multiplicities of configurations
        T - Temperature
        eci - ECI's

    Output:
        Vector representation gradient of F with Corrs
        [[d^2F/dcorr0 dcorr0, d^2F/dcorr0 dcorr1, ..., d^2F/dcorr0 dcorrn],
         [d^2F/dcorr1 dcorr0, d^2F/dcorr1 dcorr1, ..., d^2F/dcorr1 dcorrn],
         .
         .
         .
         [d^2F/dcorrn dcorr0, d^2F/dcorrn dcorr1, ..., d^2F/dcorrn dcorrn],
        ]
    """

    d2S = (np.diag(multconfig_kb / (all_vmat @ corrs)).T @ all_vmat).T @ all_vmat

    return _kB*temp*d2S
