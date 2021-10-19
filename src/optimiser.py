from scipy.optimize import minimize, basinhopping
from scipy.optimize import SR1, BFGS
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
import numpy as np
from tqdm import tqdm


def fit(F,
        vmat, kb, 
        clusters, 
        configs, configcoef,
        temp, 
        eci, 
        options,
        jac,
        hess,
        constraints,
        NUM_TRIALS,
        FIXED_CORR_1,
        num_clusters
       ):

    MIN_RES = None 
    MIN_RES_VAL = 1e5 #random large number
    bounds_corrs = Bounds([1, FIXED_CORR_1,*[-1]*(num_clusters-2)],
                          [1, FIXED_CORR_1,*[1]*(num_clusters-2)]
                         )
    for _ in tqdm(range(NUM_TRIALS)):

        corrs0 = np.array([1, FIXED_CORR_1, *np.random.uniform(-1, 1, num_clusters-2)])
#        for _ in iter(int,1): #infinite loop till a valid starting correlations are found
#            corrs0 = np.array([1, FIXED_CORR_1, *np.random.uniform(-1, 1, num_clusters-2)])
#            validcorr = np.ones(len(clusters), dtype=bool)
#
#            for cluster_idx, _ in clusters.items():
#                rho = np.matmul(vmat[cluster_idx],corrs0)
#                validcorr[cluster_idx] = np.all(rho >= 0)
#
#            if bool(np.all(validcorr)):
#                break

        temp_results = minimize(F,
                                corrs0,
                                method='trust-constr',
                                args=(vmat, kb, clusters, configs, configcoef,temp,eci),
                                options=options,
                                jac=jac,
                                hess=hess,
                                constraints=constraints,
                                bounds=bounds_corrs
                      )

        if temp_results.fun < MIN_RES_VAL:
            MIN_RES = temp_results
            MIN_RES_VAL = temp_results.fun
            tqdm.write(f"Found new minimum for x:{FIXED_CORR_1}, T:{temp} fun: {MIN_RES_VAL}")
            tqdm.write(f'Current minimum correlations: {temp_results.x}')

        for cluster_idx in clusters.keys():
            assert np.isclose(np.inner(configcoef[cluster_idx],
                                       np.matmul(vmat[cluster_idx],
                                                 temp_results.x))
                              ,1.0
                             )
    return MIN_RES
