"""
Optimiser module for SRO Correction
"""

import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import linprog
from scan_disordered import get_initial_trial


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
    obj = ecis * mults

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
        NUM_INIT_TRIALS,
        bounds,
        constraints,
        constr_tol,
        corrs_trial,
        trial_variance,
        random_trial=False,
        display_inter=False,
        approx_deriv=True,
        seed=42,
        early_stopping_cond=np.inf,
        ord2disord_dist=0.0,
        constraint=True
       ):
    """
    Functions takes in all required inputs :
        1. Functions and derivatives (if available),
        2. tolerances,
        3. bounds and constraints,
        4. Cluster Information
        4. Other fitting parameters
        and returns the minimised set of values for the particular section of the CVM correction pipeline.
    """
    rng = np.random.default_rng(seed)
    result = None
    result_value = 1e5
    constr_viol = 1e-10

    mult_arr = np.array(list(cluster_data.clustermult.values()))
    eci_arr = np.array(list(cluster_data.eci.values()))

    mults_eci = mult_arr * eci_arr

    all_vmat = np.vstack([vmat for vmat in cluster_data.vmat.values()])
    mults_config = np.array(
        list(itertools.chain.from_iterable(list(cluster_data.configmult.values()))))
    all_kb = np.array(list(itertools.chain.from_iterable([[kb for _ in range(
        len(cluster_data.configmult[idx]))] for idx, kb in cluster_data.kb.items()])))

    multconfig_kb = mults_config * all_kb

    assert all_vmat.shape == (len(multconfig_kb), len(mults_eci))

    vrhologrho = np.vectorize(lambda rho: rho * np.log(np.abs(rho)))

    if approx_deriv:
        jac = '3-point'
        hess = BFGS()

    steps_b4_mini = 0
    trial = 0
    found_optim_radius = False
    while trial < NUM_TRIALS:  # al in range(NUM_TRIALS):

        if found_optim_radius:
            if random_trial:
                corrs_trial = np.array([*corrs_trial[:len(cluster_data.single_point_clusters)+1],*rng.uniform(low=-1,high=1,size=cluster_data.num_clusters - len(cluster_data.single_point_clusters) - 1)])
                corrs_attempt = corrs_trial
            else:
                jitter = np.array([0,
                                   *[0] *
                                   len(cluster_data.single_point_clusters),
                                   *rng.normal(0,
                                               trial_variance,
                                               cluster_data.num_clusters -
                                               len(cluster_data.single_point_clusters) - 1
                                              )
                                  ]
                                 )
                corrs_attempt = corrs_trial+jitter
            fattempt = F(corrs_attempt,
                         mults_eci,
                         multconfig_kb,
                         all_vmat,
                         vrhologrho,
                         temp
                        )
        else:
            fattempt, corrs_attempt = get_initial_trial(cluster_data=cluster_data,
                                                        corr_rnd=corrs_trial,
                                                        T=temp,
                                                        ord2disord_dist=ord2disord_dist,
                                                        constraint=constraint,
                                                        trial_variance=trial_variance,
                                                        num_trials=NUM_INIT_TRIALS,
                                                        seed=seed
                                                       )

        print(f'{trial}:', end='\r')
        try:
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
        except np.linalg.LinAlgError as linalg_err:

            print(linalg_err)
            print('trying a different starting point')
            trial -= 1
            continue


        if temp_results.fun > fattempt:
            options['initial_tr_radius'] = options['initial_tr_radius']/10
            if display_inter:
                print(
                    f"Optimising Initial trust radius from {options['initial_tr_radius']*10:.2E} -->  {options['initial_tr_radius']:.2E}")
                trial -= 1


        elif temp_results.constr_violation < constr_tol and temp_results.fun < result_value: 
            found_optim_radius = True
            try:
                assert not np.all(np.isnan(temp_results.grad))
            except AssertionError:
                print('Gradient blew up!! Incorrect solution. Moving on...')
                continue

            steps_b4_mini = 0
            result = temp_results
            result_value = temp_results.fun
            if display_inter:
                print(f'Attempt Energy: {fattempt}')
                print(f'Current Energy: {temp_results.fun}')
                print(f'Current minimum correlations: {temp_results.x}')
                print(f"Gradient: {np.array2string(temp_results.grad)}")
                print(f"Constraint Violation: {temp_results.constr_violation}")
                print(f"Current Trust Radius: {temp_results.tr_radius}")
                print(
                    f"Stop Status: {temp_results.status} | {temp_results.message}")
                print('\n====================================\n')

        elif result is None:
            print('Setting results to the first optimization...')
            result = temp_results

        else:
            steps_b4_mini += 1

            if steps_b4_mini > early_stopping_cond:
                print(
                    f'No improvement for {early_stopping_cond} steps. After half of max steps ({NUM_TRIALS}) were done.')
                break


        trial += 1

    return result
