from scipy.optimize import minimize, basinhopping
from scipy.optimize import SR1, BFGS
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
        NUM_TRIALS,
        FIXED_CORR_1,
        bounds,
        constraints,
        num_clusters
       ):

    MIN_RES = None 
    MIN_RES_VAL = 1e5 #random large number
    for _ in tqdm(range(NUM_TRIALS)):

        corrs0 = np.array([1, FIXED_CORR_1, *np.random.uniform(-1, 1, num_clusters-2)])

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
            tqdm.write(f"Found new minimum for x:{FIXED_CORR_1}, T:{temp} fun: {MIN_RES_VAL}")
            tqdm.write(f'Current minimum correlations: {temp_results.x}')

        for cluster_idx in clusters.keys():
            assert np.isclose(np.inner(configcoef[cluster_idx],
                                       np.matmul(vmat[cluster_idx],
                                                 temp_results.x))
                              ,1.0
                             )
    return MIN_RES
