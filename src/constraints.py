"""
Constrain the constraints class defining the different constrains
"""
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import BFGS
from scipy.linalg import eigvals
from energyfunctions import F_hessian
class Constraints:

    def __init__(self,vmat,kb,clusters,configs,configcoef,T,eci):

        self.vmat = vmat
        self.kb = kb
        self.clusters = clusters
        self.configcoef = configcoef
        self.configs = configs
        self.T = T
        self.eci = eci

        self.constraints = []

    def set_linear_constraints(self):
        """
        constrains rhos to stay between 0 and 1 (except the 0-pt rho which are always 1)
        """
        linear_constraints = []
        for cluster_idx, _ in self.clusters.items():
            if cluster_idx == 0:
                linear_constraints.append(LinearConstraint(self.vmat[cluster_idx],
                                                           [1]*len(self.configcoef[cluster_idx]),
                                                           [1]*len(self.configcoef[cluster_idx]),
                                                           keep_feasible=True
                                                          ))
            else:
                linear_constraints.append(LinearConstraint(self.vmat[cluster_idx],
                                                           [0]*len(self.configcoef[cluster_idx]),
                                                           [1]*len(self.configcoef[cluster_idx]),
                                                           keep_feasible=True
                                                          ))

        return linear_constraints

    def constraint_singlet(self,corrs,FIXED_CORR_1):
        """
        constrains the 1-pt correlation:
        corrs[1] = FIXED_CORR_1
        """
        return corrs[1] - FIXED_CORR_1   

    def constraint_hessian(self,corrs):
        """
        constrains the hessian to be positive definite:
        by checking if all eigen values are +ve
        """
        hess = F_hessian(corrs, self.vmat, self.kb, self.clusters,
                         self.configs, self.configcoef,self.T, self.eci)
        hess_eigvals = np.real(eigvals(hess))
        return np.amin(hess_eigvals)

    def constraint_zero(self,corrs):
        """
        constrains the 1-pt correlation:
        corrs[0] = 1
        """
        return 1 - corrs[0]

    def constraint_NN(self,corrs,FIXED_CORR_2):
        """
        constrains the 2-pt correlation:
        corrs[2] = FIXED_CORR_2
        """
        return corrs[2] - FIXED_CORR_2

    def get_constraints_sro(self,FIXED_CORR_1,ord2disord_dist,corr_rnd):

        linear_constraints = self.set_linear_constraints()

        self.constraints = [*linear_constraints,
                            {'fun': self.constraint_singlet,
                             'type':'eq',
                             'args':[FIXED_CORR_1]
                            },
                            {'fun': self.constraint_zero,
                             'type':'eq'
                            },
                            {'fun': lambda x : ord2disord_dist/2 - np.linalg.norm(x-corr_rnd),
                             'type':'ineq'
                            }
                           ]
        return self.constraints

    def get_constraints_phasediagram(self,FIXED_CORR_1):

        linear_constraints = self.set_linear_constraints()

        self.constraints = [*linear_constraints,
                            {'fun': self.constraint_singlet,
                             'type':'eq',
                             'args':[FIXED_CORR_1]
                            },
                            {'fun': self.constraint_zero,
                             'type':'eq'
                            }
                           ]
        return self.constraints

    def get_constraints_hessian(self,FIXED_CORR_1,ch):

        if ch:
            linear_constraints = self.set_linear_constraints()
            self.constraints = [*linear_constraints,
                                {'fun': self.constraint_singlet,
                                 'type':'eq',
                                 'args':[FIXED_CORR_1]
                                },
                                {'fun': self.constraint_zero,
                                 'type':'eq',
                                },
                                {'fun': self.constraint_hessian ,
                                 'type':'ineq',
                                }
                               ]
        else:
            linear_constraints = self.set_linear_constraints()
            self.constraints = [*linear_constraints,
                                {'fun': self.constraint_singlet,
                                 'type':'eq',
                                 'args':[FIXED_CORR_1]
                                },
                                {'fun': self.constraint_zero,
                                 'type':'eq'
                                },
                               ]

        return self.constraints

    def get_constraints_corrscan(self,FIXED_CORR_1,FIXED_CORR_2):

        linear_constraints = self.set_linear_constraints()

        self.constraints = [*linear_constraints,
                            {'fun': self.constraint_singlet,
                             'type':'eq',
                             'args':[FIXED_CORR_1]
                            },
                            {'fun': self.constraint_NN,
                             'type':'eq',
                             'args':[FIXED_CORR_2]
                            },
                            {'fun': self.constraint_zero,
                             'type':'eq'
                            }
                           ]
        return self.constraints

    def get_constraints_muscan(self,):

        linear_constraints = self.set_linear_constraints()

        self.constraints = [*linear_constraints,
                            {'fun': self.constraint_zero,
                             'type':'eq'
                            }
                           ]

        return self.constraints

#    class MyBounds:
#        """
#        Class to constrain the trial correlations of Basin Hopping
#        """
#        def __init__(self,xmax=[1]*6, xmin=[-1]*6,):
#            self.xmax = np.array(xmax)
#            self.xmin = np.array(xmin)
#            
#        def __call__(self, **kwargs):
#            x = kwargs["x_new"]
#            tmax = bool(np.all(x <= self.xmax))
#            tmin = bool(np.all(x >= self.xmin))
#
#            return tmax and tmin
#    
#class MyTakeStep:
#    
#    def __init__(self, vmat, clusters, stepsize=0.1):
#        self.stepsize = stepsize
#        self.vmat = vmat
#        self.clusters = clusters
#        self.rng = np.random.default_rng()
#    
#    def __call__(self, x):
#        s = self.stepsize
#        
#        validcorr = np.ones(len(self.clusters), dtype=bool)
#        
#        for _ in iter(int,1):
#            x_trial = x + self.rng.uniform(-s, s, x.shape)
#            for cluster_idx, _ in self.clusters.items():
#                rho = np.matmul(self.vmat[cluster_idx],x_trial)
#                validcorr[cluster_idx] = np.all(rho >= 0)
#            if bool(np.all(validcorr)):
#                break
#            
#        return x_trial
