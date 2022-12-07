"""
Constrain the constraints class defining the different constrains
"""
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import BFGS


class Constraints:

    def __init__(self, clusterdata):
        self.clusterdata = clusterdata
        self.constraints = []
        self.all_vmat = np.vstack([vmat for vmat in clusterdata.vmat.values()])

    def set_linear_constraints(self):
        """
        constrains rhos to stay between 0 and 1 (except the 0-pt rho which are always 1)
        """
        linear_constraints = []
        linear_constraints.append({'fun': lambda x: self.all_vmat @ x,
                                   'type': 'ineq',
                                   'jac' : lambda x : self.all_vmat,
                                  }
                                 )

        return linear_constraints

    def get_constraints_sro(self, ord2disord_dist, corr_rnd, not_constrained):

        linear_constraints = self.set_linear_constraints()

        constraints_sro = [*linear_constraints,]

        if not_constrained:
            return constraints_sro

        constraints_sro.append({'fun': lambda corrs: ord2disord_dist/2 - np.linalg.norm(corrs-corr_rnd),
                                'jac': '3-point',
                                'hess': BFGS(),
                                'type': 'ineq'
                               }
                              )
        return constraints_sro

