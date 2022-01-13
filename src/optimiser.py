"""
Optimiser module for SRO Correction
"""

import numpy as np
from datetime import datetime
from scipy.linalg import eigvals
from scipy.optimize import minimize
from scipy.optimize import BFGS
from valid_corr_generator import get_valid_corrs

def fit(F,
        vmat, kb,
        clusters,
        configs, configcoef,
        temp,
        eci,
        options,
        jac,
        hess,
        NUM_TRIALS,
        FIXED_CORR_1,
        FIXED_CORR_2,
        bounds,
        constraints,
        num_clusters,
        NN,
        corrs_trial,
        ch,
        display_inter=True
       ):

    def sro_callback(xk,state):

        hess_eigvals = np.real(eigvals(hess(xk,vmat, kb, clusters, configs, configcoef,temp,eci)))
        with open(f"hessians-{int(ch)}-{datetime.now().strftime('%a')}-{datetime.now().strftime('%d')}",'a') as hess_file:
            hess_file.write(np.array2string(hess_eigvals)+'\n')
        try:
            assert all(h >= 0 for h in hess_eigvals)
        except AssertionError:
            if ch:
                print('VERY SERIOUS ERROR. Negative Hessian')
            else:
                print('WARNING: Negative Hessian')

    MIN_RES = None
    MIN_RES_VAL = 1e5 #random large number
    if NN:
        corrs0 = np.array([1, FIXED_CORR_1, FIXED_CORR_2, *np.random.uniform(-1,1,num_clusters-3)])
    else:
        corrs0 = get_valid_corrs(FIXED_CORR_1,None,vmat,clusters,num_clusters)

    for _ in range(NUM_TRIALS):

        if display_inter:
            print(f'Trial Corrs {_}: {corrs_trial}')
        jitter = np.array([0, 0, *np.random.normal(0, .001, corrs_trial[2:].shape)])
        corrs_trial = corrs_trial+jitter
        temp_results = minimize(F,
                                corrs_trial,
                                method='trust-constr',
                                args=(vmat, kb, clusters, configs, configcoef,temp,eci),
                                options=options,
                                jac=jac,
                                hess=hess,
                                constraints=constraints,
                                bounds=bounds,
#                                callback=sro_callback,
                               )

        if temp_results.fun < MIN_RES_VAL:
            MIN_RES = temp_results
            MIN_RES_VAL = temp_results.fun
            if display_inter:
                if NN:
                    print('\n')
                    print(f"Found new minimum for Corr1:{FIXED_CORR_1:.4f}, Corr2:{FIXED_CORR_2:.4f} fun: {MIN_RES_VAL:.15f}")
                else:
                    print(f"Found new minimum for x:{FIXED_CORR_1:.4f}, T:{temp} fun: {MIN_RES_VAL}")

                print(f'Current minimum correlations: {temp_results.x}')
                hessian = hess(temp_results.x, vmat, kb, clusters,
                                 configs, configcoef,temp, eci)
                hess_eigvals = np.real(eigvals(hessian))
                min_hess_eigval = np.amin(hess_eigvals)
                print(f"Eigen Values of Hessian: {hess_eigvals}")
                print(f"Gradient: {np.array2string(temp_results.grad)}")
                print(f"Stop Status: {temp_results.status} | {temp_results.message}")
                print('\n====================================\n')

    return MIN_RES

#def mufit(F,
#          vmat, kb, 
#          clusters, 
#          configs, configcoef,
#          temp, 
#          eci, 
#          options,
#          jac,
#          hess,
#          NUM_TRIALS,
#          bounds,
#          constraints,
#          num_clusters,
#          mu,
#         ):
#
#    random.seed(42)
#
#    MIN_RES = None 
#    MIN_RES_VAL = 1e5 #random large number
#    for _ in range(NUM_TRIALS):
#
#        print(_,end='\r')
#
#        corrs0 = np.array([1,*np.random.uniform(-1,1,num_clusters-1)])
#        temp_results = minimize(F,
#                                corrs0,
#                                method='trust-constr',
#                                args=(vmat, kb, clusters, configs, configcoef,temp,eci,mu),
#                                options=options,
#                                jac=jac,
#                                hess=hess,
#                                constraints=constraints,
#                                bounds=bounds
#                               )
#
#        if temp_results.fun < MIN_RES_VAL:
#            MIN_RES = temp_results
#            MIN_RES_VAL = temp_results.fun
#
#            print(f"Found new minimum for Mu :{mu}, fun: {MIN_RES_VAL:.15f}")
#            print(f'Current minimum correlations: {temp_results.x}')
#            print("Rhos:")
#            for val in temp_results.constr[:num_clusters]:
#                print(np.array2string(val))
#            print(f"Gradient: {np.array2string(temp_results.grad)}")
#            print(f"Stop Status: {temp_results.status} | {temp_results.message}")
#            print('\n====================================\n')
#
#    return MIN_RES
