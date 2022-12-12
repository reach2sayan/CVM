"""
Constrain the constraints class defining the different constrains
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import BFGS

@dataclass(frozen=True)
class Constraints:
    _all_vmat: np.ndarray
    _ordered_correlations: np.ndarray = None
    _disordered_correlations: np.ndarray = None
    _norm_constrained: bool = False

    _linear_constraints: list[dict] = field(init=False,default_factory=list)
    _norm_constraints: list[dict] = field(init=False,default_factory=list)

    def __post_init__(self: Constraints) -> None:

        object.__setattr__(self,
                           '_linear_constraints',
                           [{'fun': lambda x: self._all_vmat @ x,
                            'type': 'ineq',
                            'jac' : lambda x : self._all_vmat,
                           }]
                          )
        if self._norm_constrained:
            object.__setattr__(self,
                               '_norm_constraints',
                               [{'fun': lambda x: np.linalg.norm(self._ordered_correlations - self._disordered_correlations)/2 - np.linalg.norm(x-self._disordered_correlations),
                                'type': 'ineq',
                                'jac': '3-point',
                                'hess': BFGS(),
                               }]
                              )

    @property
    def constraints(self):
       return [*self._linear_constraints, *self._norm_constraints]
