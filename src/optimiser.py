"""
Optimiser module for SRO Correction
"""

import itertools
import numpy as np
from scipy.optimize import minimize, basinhopping, shgo
from scipy.optimize import BFGS
from scipy.optimize import linprog
from scan_disordered import get_random_structure
import subprocess
from basinhopping_features import BasinHoppingBounds, BasinHoppingStep, BasinHoppingCallback


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

def fit_sro_correction_basinhopping(F,
                                    cluster_data,
                                    temp,
                                    options,
                                    jac,
                                    hess,
                                    NUM_TRIALS,
                                    bounds,
                                    constraints,
                                    constr_tol,
                                    corrs_trial,
                                    trial_variance,
                                    structure,
                                    lattice_file,
                                    clusters_file,
                                    display_inter=False,
                                    random_trial=False,
                                    approx_deriv=True,
                                    seed=42,
                                    early_stopping_cond=np.inf,
                                    ord2disord_dist=0.0,
                                    num_atoms_per_clus=1,
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

    mult_arr = np.array(list(cluster_data.clustermult.values()))
    eci_arr = np.array(list(cluster_data.eci.values()))

    mults_eci = mult_arr * eci_arr

    all_vmat = np.vstack(list(cluster_data.vmat.values()))
    mults_config = np.array(
        list(itertools.chain.from_iterable(list(cluster_data.configmult.values()))))
    all_kb = np.array(list(itertools.chain.from_iterable([[kb for _ in range(
        len(cluster_data.configmult[idx]))] for idx, kb in cluster_data.kb.items()])))

    multconfig_kb = mults_config * all_kb

    assert all_vmat.shape == (len(multconfig_kb), len(mults_eci))

    vrhologrho = np.vectorize(lambda rho: rho * np.log(np.abs(rho)))
    #vrhologrho = np.vectorize(lambda rho: rho * np.log(rho))

    if approx_deriv:
        jac = '3-point'
        hess = BFGS()

    ff_trial = F(corrs_trial,
                 mults_eci,
                 multconfig_kb,
                 all_vmat,
                 vrhologrho,
                 temp
                )

    result_value = ff_trial
    result_constr_viol = np.float64(0.0)
    result_corr = corrs_trial.copy()
    result_grad = np.zeros(corrs_trial.shape[0])

    args = (
        mults_eci,
        multconfig_kb,
        all_vmat,
        vrhologrho,
        temp,
    )

    minimizer_kwargs = {'args' : args,
                        'method': 'trust-constr',
                        'options': options,
                        'jac': jac, 'hess': hess,
                        'constraints' : constraints,
                        'bounds': bounds,
                       }

    bh_step = BasinHoppingStep(e_F = F,
                               mults_eci = mults_eci,
                               multconfig_kb = multconfig_kb,
                               all_vmat = all_vmat,
                               vrhologrho = vrhologrho,
                               temp = temp,
                               cluster_data = cluster_data,
                               structure = structure,
                               num_atoms_per_clus = num_atoms_per_clus,
                               lattice_file = lattice_file,
                               clusters_file = clusters_file,
                               stepsize = trial_variance,
                               seed = seed,
                               display_inter = display_inter,
                               random_trial = random_trial,
                              )

    bh_accept = BasinHoppingBounds(all_vmat)

    bh_callback = BasinHoppingCallback(cluster_data, temp, num_atoms_per_clus, corrs_trial)

    result = basinhopping(F,
                          x0=corrs_trial,
                          minimizer_kwargs = minimizer_kwargs,
                          niter = NUM_TRIALS, 
                          stepsize = trial_variance,
                          take_step = bh_step,
                          accept_test = bh_accept,
                          niter_success = early_stopping_cond,
                          interval = 5,
                          seed = seed,
                          callback = bh_callback,
                          disp=True
                         )

    try:
        assert result.lowest_optimization_result.constr_violation < constr_tol
    except AssertionError:
        print('Major Constrain violation')

    result_value = result.fun
    result_grad = result.lowest_optimization_result.grad.copy()
    result_corr = result.x.copy()
    result_constr_viol = result.lowest_optimization_result.constr_violation

    return result_value, result_corr, result_grad, result_constr_viol

def fit_sro_correction(F,
                       cluster_data,
                       temp,
                       options,
                       jac,
                       hess,
                       NUM_TRIALS,
                       bounds,
                       constraints,
                       constr_tol,
                       corrs_trial,
                       trial_variance,
                       structure,
                       lattice_file,
                       clusters_file,
                       random_trial=False,
                       display_inter=False,
                       approx_deriv=True,
                       seed=42,
                       early_stopping_cond=np.inf,
                       ord2disord_dist=0.0,
                       num_atoms_per_clus=1,
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
    randgen = get_random_structure(structure)

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

    earlystop = 0
    trial = 0

    first_trial = corrs_trial.copy()
    ff_trial = F(first_trial,
                 mults_eci,
                 multconfig_kb,
                 all_vmat,
                 vrhologrho,
                 temp
                )

    result_value = ff_trial
    result_constr_viol = np.float64(0.0)
    result_corr = first_trial.copy()
    result_grad = np.zeros(first_trial.shape[0])

    while trial < NUM_TRIALS:  # al in range(NUM_TRIALS):

        accepted = False
        if display_inter:
            print(f'Trial No.: {trial}')
        if random_trial:
            next(randgen)
            corrs_attempt = subprocess.run(['corrdump', '-c', f'-cf={structure}/{clusters_file}', f'-s={structure}/randstr.in', f'-l={structure}/{lattice_file}'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           check=True
                                          )
            # convert from bytes to string list
            corrs_attempt = corrs_attempt.stdout.decode('utf-8').split('\t')[:-1]
            corrs_attempt = np.array(corrs_attempt, dtype=np.float32)  # convert to arrays
            assert cluster_data.check_result_validity(corrs_attempt)

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

        if trial == 0:
            corrs_attempt = first_trial

        fattempt = F(corrs_attempt,
                     mults_eci,
                     multconfig_kb,
                     all_vmat,
                     vrhologrho,
                     temp
                    )

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

        if temp_results is None:
            print('WARNING: Optimisation failed')

        elif temp_results.constr_violation < constr_tol and temp_results.fun < result_value:

            try:
                assert not np.all(np.isnan(temp_results.grad))
            except AssertionError:
                print('Gradient blew up!! Incorrect solution. Moving on...')
                continue

            if temp_results.fun > fattempt:
                print('WARNING. Final energy higher than trial correlation')

            earlystop = 0
            result_value = temp_results.fun
            result_grad = temp_results.grad.copy()
            result_corr = temp_results.x.copy()
            result_constr_viol = temp_results.constr_violation

            accepted = True

        earlystop += 1

        if earlystop > early_stopping_cond and trial > NUM_TRIALS/2:
            print(
                f'No improvement for consecutive {early_stopping_cond} steps. After half of total steps ({int(NUM_TRIALS/2)}) were done')
            break
        trial += 1

        if display_inter and temp_results is not None:
            print(f'Current attempt correlations: {corrs_attempt}')
            print(f'Trial Validity: {cluster_data.check_result_validity(corrs_attempt)}')
            print(f'Attempt Free Energy @ T = {temp}K : {fattempt/num_atoms_per_clus}')
            print(f'Current Free Energy @ T = {temp}K : {temp_results.fun/num_atoms_per_clus}')
            print(f'Current minimum correlations: {temp_results.x}')
            print(f"Gradient: {np.array2string(temp_results.grad)}")
            print(f"Constraint Violation: {temp_results.constr_violation}")
            print(f"Constraint Penalty: {temp_results.constr_penalty}")
            print(f"Barrier Parameter: {temp_results.barrier_parameter}")
            print(f"Barrier Tolerance: {temp_results.barrier_tolerance}")
            print(f"Current Trust Radius: {temp_results.tr_radius}")
            print(
                f"Stop Status: {temp_results.status} | {temp_results.message}")
            print(f"Acccepted? : {accepted}")
            print(f"Current Min Free Energy @ T = {temp}K : {result_value/num_atoms_per_clus}")
            print(f"Current Best Contr. Viol : {result_constr_viol}")
            print(f"Fully Disordered? : {np.allclose(result_corr,first_trial)}")
            print('\n====================================\n')

    return result_value, result_corr, result_grad, result_constr_viol
