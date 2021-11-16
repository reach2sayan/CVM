import numpy as np

def get_valid_corrs(FIXED_CORR1,FIXED_CORR2,vmat,clusters,num_clusters):

    for _ in iter(int,1):
        corr1 = FIXED_CORR1
        validcorr = np.ones(num_clusters, dtype=bool)
        if corr1 <=0 :
            if FIXED_CORR2 is None:
                corr2 = np.random.uniform(-2*corr1 - 1, 1)
                corr3 = np.random.uniform(-2*corr1 - 1, 1)
                corr4 = np.random.uniform(corr1+corr2-1,corr1-corr2+1)
                corr5 = np.random.uniform(2*corr3-1,1)
            else:
                corr2 = FIXED_CORR2
                corr3 = FIXED_CORR2
                corr4 = np.random.uniform(-1,1)
                corr5 = np.random.uniform(-1,1)
        else:
            if FIXED_CORR2 is None:
                corr2 = corr2 = np.random.uniform(2*corr1 - 1, 1)
                corr3 = np.random.uniform(2*corr1 - 1, 1)
                corr4 = np.random.uniform(corr1+corr2-1,corr1-corr2+1)
                corr5 = np.random.uniform(2*corr3-1,1)
            else:
                corr2 = FIXED_CORR2
                corr3 = FIXED_CORR2
                corr4 = np.random.uniform(-1,1)
                corr5 = np.random.uniform(-1,1)

        corrs0 = np.array([1, corr1, corr2, corr3, corr4, corr5])

        for cluster_idx, _ in clusters.items():
            rho = np.matmul(vmat[cluster_idx],corrs0)
            validcorr[cluster_idx] = np.all(rho >= 0)
            
        if bool(np.all(validcorr)):
            break

    return corrs0
