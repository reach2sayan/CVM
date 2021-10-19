import numpy as np

def constraint_rhos_sum(corrs, vmat, clusters, configcoef,):
    """
    Constraints the sum of each rho. As of now, it's done in a weird way, where the total sum of the array:
    [1 - sum(rho), .... ] is constrained to sum to 0. This along with the constraint that each rho is between
    0 and 1, seems to make it work. I think that by the this might be a redundant constraint as well.
    """
    rho_sum = []

    def clus_prob(cluster_idx):
        rho = np.matmul(vmat[cluster_idx],corrs)
        return rho
    
    for cluster_idx, _ in clusters.items():
        rho = clus_prob(cluster_idx)
        rho_sum.append(np.sum(configcoef[cluster_idx]*rho))
    
    return np.sum(1 - np.array(rho_sum))

def constraint_singlet(corrs,FIXED_CORR_1):
    """
    constrains the 1-pt correlation:
    corrs[1] = FIXED_CORR_1
    """
    return corrs[1] - FIXED_CORR_1   

def constraint_zero(corrs):
    """
    constrains the 1-pt correlation:
    corrs[0] = 1
    """
    return 1 - corrs[0]

def constraint_NN(corrs,FIXED_CORR_2):
    """
    constrains the 2-pt correlation:
    corrs[2] = FIXED_CORR_2
    """
    return corrs[2] - FIXED_CORR_2 

class MyBounds:
    """
    Class to constrain the trial correlations of Basin Hopping
    """
    def __init__(self,xmax=[1]*6, xmin=[-1]*6,):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
        
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))

        return tmax and tmin
    
class MyTakeStep:
    
    def __init__(self, vmat, clusters, stepsize=0.1):
        self.stepsize = stepsize
        self.vmat = vmat
        self.clusters = clusters
        self.rng = np.random.default_rng()
    
    def __call__(self, x):
        s = self.stepsize
        
        validcorr = np.ones(len(self.clusters), dtype=bool)
        
        for _ in iter(int,1):
            x_trial = x + self.rng.uniform(-s, s, x.shape)
            for cluster_idx, _ in self.clusters.items():
                rho = np.matmul(self.vmat[cluster_idx],x_trial)
                validcorr[cluster_idx] = np.all(rho >= 0)
            if bool(np.all(validcorr)):
                break
            
        return x_trial
