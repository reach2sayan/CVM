"""
Optimiser module for SRO Correction
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import linprog
import itertools


def find_ordered(cluster_data, corr, method, options, fix_point=True, print_output=True,):
    """
    Finds the ordered structure given cluster information using Linear Programming
    Input:
        cluster_data - ClusterInfo object contatning vmat, eci and cluster information
        corr - sample corr, only used as a guess for the refined simplex method. This also
               specified the point correlations that are kept fixed.
        method - method of the linear programming; simplex or interior point
        options - extra options for the linear programming problem
    Output
    """

    all_vmat = -1 * np.vstack([vmat for vmat in cluster_data.vmat.values()])
    vmat_limit = np.zeros(all_vmat.shape[0])
    if fix_point:
        corr_bounds = [(corr[idx], corr[idx]) if cluster['type'] == 1 else (
            1, 1) if cluster['type'] == 0 else (-1, 1) for idx, cluster in cluster_data.clusters.items()]
    else:
        if cluster_data.clusters is None:
            corr_bounds = [(-1, 1) for _ in cluster_data.eci]
        else:
            corr_bounds = [(1, 1) if cluster['type'] == 0 else (-1, 1)
                           for idx, cluster in cluster_data.clusters.items()]

    ecis = np.array(list(cluster_data.eci.values())),
    mults = np.array(list(cluster_data.clustermult.values()))
    obj = ecis*mults

    result = linprog(obj,
                     A_ub=all_vmat,
                     b_ub=vmat_limit,
                     bounds=corr_bounds,
                     options=options,
                     method=method
                     )
    if result.success:
        print('Ordered State calculations completed...')
        if print_output:
            np.savetxt('ordered_correlations.out', result.x)
            with open('ordered_rho.out', 'w') as frho:
                for vmat in cluster_data.vmat.values():
                    frho.write(f'{" ".join(map(str,vmat@result.x))}\n')

        return result
    print(
        f'WARNING: linear programming for ordered correlation search failed: {result.status} - {result.message}\nExiting...')
    return result


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
        seed=42,
        early_stopping_cond=np.inf
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
    mult_arr = np.array(list(cluster_data.clustermult.values()))
    eci_arr = np.array(list(cluster_data.eci.values()))

    all_vmat = np.vstack([vmat for vmat in cluster_data.vmat.values()])
    mults_config = np.array(
        list(itertools.chain.from_iterable(list(cluster_data.configmult.values()))))
    mults_eci = np.multiply(mult_arr, eci_arr)
    all_kb = np.array(list(itertools.chain.from_iterable([[kb for _ in range(
        len(cluster_data.configmult[idx]))] for idx, kb in cluster_data.kb.items()])))

    multconfig_kb = np.multiply(mults_config, all_kb)

    def rhologrho(rho): return rho * np.log(np.abs(rho))
    vrhologrho = np.vectorize(rhologrho)

    if approx_deriv:
        print('Approximating the derivatives - Jacobian : a 3-point finite diffrence scheme, Hessian : BFGS')
        jac = '3-point'
        hess = BFGS()
    else:
        print('Using Analytical gradients and hessian...')

    steps_b4_mini = 0
    for trial in range(NUM_TRIALS):
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

        #print(f'{trial} : {corrs_attempt}',end='\r')
        temp_results = minimize(F,
                                corrs_attempt,
                                method='trust-constr',
                                args=(
                                    mults_eci,
                                    multconfig_kb,
                                    all_vmat,
                                    vrhologrho,
                                    temp,
                                ),
                                options=options,
                                jac=jac,
                                hess=hess,
                                constraints=constraints,
                                bounds=bounds,
                                )

        if temp_results.fun < result_value:
            steps_b4_mini = 0
            result = temp_results
            result_value = temp_results.fun
            if display_inter:
                print(f'Current Energy: {temp_results.fun}')
                print(f'Current minimum correlations: {temp_results.x}')
                print(f"Gradient: {np.array2string(temp_results.grad)}")
                print(
                    f"Stop Status: {temp_results.status} | {temp_results.message}")
                print('\n====================================\n')
        else:
            steps_b4_mini += 1

        if trial > NUM_TRIALS/2 and steps_b4_mini > early_stopping_cond:
            print(
                f'No improvement for {early_stopping_cond} steps. After half of max steps ({NUM_TRIALS}) were done.')
            break

    return result
