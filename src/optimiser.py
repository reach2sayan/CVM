from scipy.optimize import minimize, basinhopping
from scipy.optimize import SR1, BFGS
from valid_corr_generator import get_valid_corrs
import numpy as np
import random

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
        NN
       ):

    random.seed(42)

    MIN_RES = None 
    MIN_RES_VAL = 1e5 #random large number
    for _ in range(NUM_TRIALS):

        print(_,end='\r')

        if NN:
            corrs0 = np.array([1,FIXED_CORR_1,FIXED_CORR_2, *np.random.uniform(-1,1,num_clusters-3)])
#            corrs0 = get_valid_corrs(FIXED_CORR_1,FIXED_CORR_2,vmat,clusters,num_clusters)
        else:
            corrs0 = get_valid_corrs(FIXED_CORR_1,None,vmat,clusters,num_clusters)

        temp_results = minimize(F,
                                corrs0,
                                method='trust-constr',
                                args=(vmat, kb, clusters, configs, configcoef,temp,eci),
                                options=options,
                                jac=jac,
                                hess=hess,
                                constraints=constraints,
                                bounds=bounds
                      )

        if temp_results.fun < MIN_RES_VAL:
            MIN_RES = temp_results
            MIN_RES_VAL = temp_results.fun
            if NN:
                print('\n')
                print(f"Found new minimum for Corr1:{FIXED_CORR_1:.4f}, Corr2:{FIXED_CORR_2:.4f} fun: {MIN_RES_VAL:.15f}")
            else:
                print(f"Found new minimum for x:{FIXED_CORR_1:.4f}, T:{temp} fun: {MIN_RES_VAL}")

            print(f'Current minimum correlations: {temp_results.x}')
            print("Rhos:")
            for val in temp_results.constr[:num_clusters]:
                print(np.array2string(val))
            print(f"Gradient: {np.array2string(temp_results.grad)}")
            print(f"Stop Status: {temp_results.status} | {temp_results.message}")
            print('\n====================================\n')

    return MIN_RES
