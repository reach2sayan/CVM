import numpy as np
import math
import sys

kB = 8.617330337217213e-05
def F_efficient(corrs, mults_eci, multconfig_kb, all_vmat, vect_rhologrho, temp):

    H = mults_eci @ corrs
    S = multconfig_kb @ vect_rhologrho((all_vmat @ corrs) + sys.float_info.epsilon)

    return H + kB*temp*S

def F_jacobian_efficient(corrs, mults_eci, multconfig_kb, all_vmat, vect_rhologrho, temp):

    dH = mults_eci
    dS = all_vmat.T @ (multconfig_kb * (1 + np.log(np.abs((all_vmat @ corrs) + sys.float_info.epsilon))))
#    dS = all_vmat.T @ (multconfig_kb * (1 + np.log((all_vmat @ corrs) + sys.float_info.epsilon)))

    return dH + kB*temp*dS

def F_hessian_efficient(corrs, mults_eci, multconfig_kb, all_vmat, vect_rhologrho, temp):

    #d2S = ((multconfig_kb / (all_vmat @ corrs))[:, np.newaxis] * all_vmat).T @ all_vmat
    d2S = (np.diag(multconfig_kb / (all_vmat @ corrs)).T @ all_vmat).T @ all_vmat

    return kB*temp*d2S

def F(corrs, vmat, kb, clusters, clustermult, configmult, T, eci):
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

    def get_corrsum(vmat, corrs):
        assert len(vmat) == len(corrs)
        corrsum = np.inner(vmat, corrs)
        if corrsum == 0:
            return 0
        return corrsum*math.log(np.abs(corrsum))

    def per_cluster_sum(corrs, vmat, configmult):
        config_sum = np.sum([coef*get_corrsum(vmat[config_idx], corrs)
                             for config_idx, coef in enumerate(configmult)
                             ])
        return config_sum

    H = np.sum([clustermult[cluster_idx]*eci[cluster_idx]*corrs[cluster_idx]
                for cluster_idx in clusters
                ])
    S = np.sum([kb[config_idx]*per_cluster_sum(corrs,
                                               vmat[config_idx],
                                               configmult[config_idx],)
                for config_idx in configmult
                ])

    return H + kB*T*S


def F_jacobian(corrs, vmat, kb, clusters, clustermult, configmult, T, eci):
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

    def get_kth_elem_jac(corrs, vmat, kb, configmult, corr_idx):

        dS_k = np.sum([kb[config_idx] * per_cluster_sum_jac(corrs,
                                                            vmat[config_idx],
                                                            configmult[config_idx],
                                                            corr_idx,
                                                            )
                       for config_idx in configmult
                       ])
        return dS_k

    def get_corrsum_jac(vmat, corrs):
        assert len(vmat) == len(corrs)
        corrsum = np.inner(vmat, corrs)
        return 1 + math.log(np.abs(corrsum))

    def per_cluster_sum_jac(corrs, vmat, configmult, corr_idx):

        config_sum = np.sum([coef * vmat[config_idx][corr_idx] * get_corrsum_jac(vmat[config_idx], corrs)
                             for config_idx, coef in enumerate(configmult)
                             ])
        return config_sum

    dH = np.array([clustermult[cluster_idx]*eci[cluster_idx]
                   for cluster_idx in clusters
                   ])
    dS = np.array([get_kth_elem_jac(corrs, vmat, kb, configmult, corr_idx)
                   for corr_idx, _ in enumerate(corrs)
                   ])

    return dH + kB*T*dS




def F_hessian(corrs, vmat, kb, clusters, clustermult, configmult, T, eci):
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

    def get_corrsum_hess(vmat, corrs):

        assert len(vmat) == len(corrs)
        corrsum = np.inner(vmat, corrs)
        return corrsum

    def get_config_val(corrs, vmat, configmult, corr_idx_1, corr_idx_2):

        config_val = np.sum([coef * vmat[config_idx][corr_idx_1] * vmat[config_idx][corr_idx_2] / get_corrsum_hess(vmat[config_idx], corrs)
                             for config_idx, coef in enumerate(configmult)
                             ])

        return config_val

    def get_hessian_elem(corrs, vmat, kb, configmult, corr_idx_1, corr_idx_2):

        hess_elem = np.sum([kb[config_idx] * get_config_val(corrs,
                                                            vmat[config_idx],
                                                            configmult[config_idx],
                                                            corr_idx_1,
                                                            corr_idx_2
                                                            )
                            for config_idx in configmult
                            ])
        return hess_elem

    d2F = np.empty([len(corrs), len(corrs)])

    d2F = np.array([[get_hessian_elem(corrs, vmat, kb, configmult, corr_idx_1, corr_idx_2)
                     for corr_idx_2, _ in enumerate(corrs)] for corr_idx_1, _ in enumerate(corrs)])

    return kB*T*d2F

