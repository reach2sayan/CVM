"""
Optimiser module for SRO Correction
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import BFGS


def fit(F,
        cluster_data,
        temp,
        options,
        jac,
        hess,
        NUM_TRIALS,
        bounds,
        constraints,
        corrs_trial,
        display_inter=False,
        approx_deriv=True,
        init_random=False,
        init_disordered=True,
        seed=42
        ):
    """
    Functions takes in all required inputs :
        1. Functions and derivatives (if available),
        2. tolerances,
        3. bounds and constraints,
        4. Other fitting parameters
    and returns the minimised set of values for the particular section of the CVM correction pipeline.
    """
    rng = np.random.default_rng(seed)
    result = None
    result_value = 1e5
    if approx_deriv:
        jac = '3-point'
        hess = BFGS()

    for _ in range(NUM_TRIALS):
        if init_random:
            corrs_attempt = np.array([1, *[corrs_trial[1]]*len(cluster_data.single_point_clusters),
                                      *rng.uniform(-1, 1, cluster_data.num_clusters - len(cluster_data.single_point_clusters) - 1)
                                      ]
                                     )
        elif init_disordered:
            jitter = np.array([0,
                               *[0]*len(cluster_data.single_point_clusters),
                               *rng.normal(0, .001, cluster_data.num_clusters - len(cluster_data.single_point_clusters) - 1)
                               ]
                              )
            corrs_attempt = corrs_trial+jitter

        #print(f'{_} : {corrs_attempt}',)
        temp_results = minimize(F,
                                corrs_attempt,
                                method='trust-constr',
                                args=(cluster_data.vmat,
                                      cluster_data.kb,
                                      cluster_data.clusters,
                                      cluster_data.configcoef,
                                      temp,
                                      cluster_data.eci),
                                options=options,
                                jac=jac,
                                hess=hess,
                                constraints=constraints,
                                bounds=bounds,
                                )

        if temp_results.fun < result_value:
            result = temp_results
            result_value = temp_results.fun
            if display_inter:
                print(f'Current Energy: {temp_results.fun}')
                print(f'Current minimum correlations: {temp_results.x}')
                print(f"Gradient: {np.array2string(temp_results.grad)}")
                print(
                    f"Stop Status: {temp_results.status} | {temp_results.message}")
                print('\n====================================\n')

    return result
