"""
Constrain the constraints class defining the different constrains
"""
import numpy as np
from scipy.optimize import LinearConstraint

class Constraints:

    def __init__(self,clusterdata):
        self.clusterdata = clusterdata
        self.constraints = []

    def constraint_singlet(self,corrs,FIXED_CORR, index):
        """
        constrains the 1-pt correlation:
            corrs[1] = FIXED_CORR_1
        """
        return corrs[index] - FIXED_CORR[index]

    def constraint_zero(self,corrs):
        """
        constrains the 1-pt correlation:
        corrs[0] = 1
        """
        return 1 - corrs[0]

    def set_linear_constraints(self):
        """
        constrains rhos to stay between 0 and 1 (except the 0-pt rho which are always 1)
        """
        linear_constraints = []
        for config_idx, _ in self.clusterdata.configcoef.items():
            if config_idx == 0:
                linear_constraints.append(LinearConstraint(self.clusterdata.vmat[config_idx],
                                                                [1]*len(self.clusterdata.configcoef[config_idx]),
                                                                [1]*len(self.clusterdata.configcoef[config_idx]),
                                                               )
                                              )
            else:
                linear_constraints.append(LinearConstraint(self.clusterdata.vmat[config_idx],
                                                                [0]*len(self.clusterdata.configcoef[config_idx]),
                                                                [1]*len(self.clusterdata.configcoef[config_idx]),
                                                               )
                                              )

        return linear_constraints

    def set_singlet_constraints(self, corr_rnd):
        """
        constrains all point correlations to conserve composition
        """

        singlet_constraints = []
        for cluster_idx, cluster in self.clusterdata.clusters.items():
            if cluster['type'] == 1:
                singlet_constraints.append({'fun': self.constraint_singlet,
                                        'type':'eq',
                                        'args':[corr_rnd, cluster_idx]
                                       }
                                      )

        return singlet_constraints

    def get_constraints_sro(self, ord2disord_dist, corr_rnd):

        linear_constraints = self.set_linear_constraints()
        singlet_constraints = self.set_singlet_constraints(corr_rnd)

        constraints_sro = [*linear_constraints,
                           *singlet_constraints,
                           {'fun': self.constraint_zero,
                            'type':'eq'
                           },
                           {'fun': lambda corrs : ord2disord_dist/2 - np.linalg.norm(corrs-corr_rnd),
                            'type':'ineq'
                           }
                          ]
        return constraints_sro


    def get_constraints_ordered(self,corr_rnd):

        linear_constraints = self.set_linear_constraints()
        singlet_constraints = self.set_singlet_constraints(corr_rnd)
        constraints_ordered = [*linear_constraints,
                               *singlet_constraints,
                               {'fun': self.constraint_zero,
                                'type':'eq'
                               },
                              ]
        return constraints_ordered
