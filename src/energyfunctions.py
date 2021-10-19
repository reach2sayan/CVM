import numpy as np
import math
from ase.units import kB

def F(corrs, vmat, kb, clusters, configs, configcoef,T,eci):
    """
    Input: 
    corrs - Correlations
    vmat  - V-Matrix
    clusters - Maximal Cluster Information (multiplicity, longest neighbor length, no. of points)
    configs - Not used
    configcoef - Coefficients of subclusters - array containing the coeff of each subcluster
    T - Temperature
    eci - ECI's
    
    Output:
    F = H + kB*T*SUM(rho * log(rho))
    """
    
    def get_corrsum(vmat,corrs):
        assert len(vmat) == len(corrs)
        corrsum = np.inner(vmat,corrs)
        if corrsum == 0:
            return 0
            #corrsum = np.finfo(float).tiny

        return corrsum * math.log(np.abs(corrsum))
    
    def per_cluster_sum(corrs,vmat,configcoef):
        config_sum = np.sum([coef * get_corrsum(vmat[config_idx],corrs) for config_idx, coef in enumerate(configcoef)
                            ])
                      
        return config_sum
    
    H = np.sum([cluster['mult']*eci[cluster_idx]*corrs[cluster_idx] 
                for cluster_idx, cluster in clusters.items()
               ])
    
    S = np.sum([kb[cluster_idx]*per_cluster_sum(corrs,
                                                vmat[cluster_idx],
                                                configcoef[cluster_idx],)
                for cluster_idx in clusters.keys()
               ])
    
    return H + kB*T*S

def F_jacobian(corrs, vmat, kb, clusters, configs, configcoef,T,eci):
    """
    Input: 
    corrs - Correlations
    vmat  - V-Matrix
    clusters - Maximal Cluster Information (multiplicity, longest neighbor length, no. of points)
    configs - Not used
    configcoef - Coefficients of subclusters - array containing the coeff of each subcluster
    T - Temperature
    eci - ECI's
    
    Output:
    Vector representation gradient of F with Corrs
    [dF/dcorr0, dF/dcorr1, ...]
    """
    
    def get_kth_elem_jac(corrs, vmat, kb, clusters, configs, configcoef,corr_idx):
        
        dS_k = np.sum([kb[cluster_idx]*per_cluster_sum_jac(corrs,
                                                           vmat[cluster_idx],
                                                           configcoef[cluster_idx],
                                                           corr_idx,) 
                       for cluster_idx in clusters.keys()
                      ])
        return dS_k
    
    def get_corrsum_jac(vmat,corrs):
        assert len(vmat) == len(corrs)
        corrsum = np.inner(vmat,corrs)
        if corrsum == 0:
            return np.NINF
            #corrsum = np.finfo(float).tiny

        return 1 + math.log(np.abs(corrsum))

    def per_cluster_sum_jac(corrs,vmat,configcoef,corr_idx):
        
        config_sum = np.sum([coef * vmat[config_idx][corr_idx] * get_corrsum_jac(vmat[config_idx],corrs) for config_idx, coef in enumerate(configcoef)
                            ])

        
        return config_sum
    
    dH = np.array([cluster['mult']*eci[cluster_idx] for cluster_idx, cluster in clusters.items()])
    
    dS = np.array([get_kth_elem_jac(corrs, vmat, kb, clusters, configs, configcoef,corr_idx) for corr_idx, _ in enumerate(corrs)])
    
    return dH + kB*T*dS

def F_hessian(corrs, vmat, kb, clusters, configs, configcoef,T,eci):
    """
    Input:
    corrs - Correlations
    vmat  - V-Matrix
    clusters - Maximal Cluster Information (multiplicity, longest neighbor length, no. of points)
    configs - Not used
    configcoef - Coefficients of subclusters - array containing the coeff of each subcluster
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

    def get_corrsum_hess(vmat,corrs):
        assert len(vmat) == len(corrs)

        corrsum = np.inner(vmat,corrs)
        if corrsum == 0:
            return 1/np.PINF
            #corrsum = np.finfo(float).tiny

        return corrsum

    def get_config_val(corrs,vmat,configcoef,corr_idx_1,corr_idx_2):

        config_val = np.sum([coef * vmat[config_idx][corr_idx_1] * vmat[config_idx][corr_idx_2] / get_corrsum_hess(vmat[config_idx],corrs)
                             for config_idx, coef in enumerate(configcoef)
                            ])

        return config_val

    def get_hessian_elem(corrs, vmat, kb, clusters, configs, configcoef,T,eci,corr_idx_1,corr_idx_2):

        hess_elem = np.sum([kb[cluster_idx] * get_config_val(corrs,
                                                             vmat[cluster_idx],
                                                             configcoef[cluster_idx],
                                                             corr_idx_1,
                                                             corr_idx_2
                                                            )
                            for cluster_idx in clusters.keys()
                           ])
        return hess_elem

    d2F = np.empty([len(corrs),len(corrs)])

    d2F = np.array([[get_hessian_elem(corrs, vmat, kb, clusters, configs, configcoef, T, eci, corr_idx_1, corr_idx_2) for corr_idx_2, _ in enumerate(corrs)] for corr_idx_1, _ in enumerate(corrs)])

    return d2F