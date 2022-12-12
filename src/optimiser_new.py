"""
Optimiser module for SRO Correction
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Type
import subprocess

import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.optimize import BFGS
from scipy.optimize import linprog
from clusterdata_new import Cluster

from energyfunctions_new import F, F_jacobian, F_hessian
from constraints_new import Constraints
from bounds_new import CorrBounds

from atatio import get_random_structure
from basinhopping_features import BasinHoppingBounds, BasinHoppingStep, BasinHoppingCallback

@dataclass(repr=False)
class ClusterOptimizer:

    cluster: Type[Cluster]
    print_output: bool = True
    approx_deriv: bool = True

    _F: Callable[[np.ndarray,
                  np.ndarray,
                  np.ndarray,
                  np.ndarray,
                  Callable[[np.ndarray],np.ndarray],
                  float,
                 ], float] = F
    _vrhologrho: Callable[[np.ndarray],np.ndarray] = np.vectorize(lambda rho: rho * np.log(np.abs(rho)))
    _seed: int = 42,
    _dF: Callable[...,np.ndarray] = field(init=False)
    _d2F: Callable[...,np.ndarray] = field(init=False)
    _num_lat_atoms: int

    def __post_init__(self):
        print('inside post-init')
        if self.approx_deriv:
            self._dF = '3-point'
            self._d2F = BFGS()
        else:
            self._dF = F_jacobian
            self._d2F = F_hessian

@dataclass
class OrderedStateOptimizer(ClusterOptimizer):
    """
    Finds the ordered structure given cluster information using Linear Programming
    Input:
        cluster_data - ClusterInfo object contatning vmat, eci and cluster information
        corr - sample corr, only used as a guess for the refined simplex method. This also
        specified the point correlations that are kept fixed.
        method - method of the linear programming; simplex or interior point
        options - extra options for the linear programming problem
        Output
    """

    options: dict
    method: str = 'revised-simplex'

    def fit(self, initial_correlation):

        all_vmat_neg = -1 * self.cluster.vmatrix_array
        vmat_limit = np.zeros(all_vmat_neg.shape[0])
        corr_bounds = [(initial_correlation[idx], initial_correlation[idx]) if cluster['type'] == 1 else (
            1, 1) if cluster['type'] == 0 else (-1, 1) for idx, cluster in self.cluster._clusters.items()]
        obj = self.cluster.eci_array * self.cluster.clusmult_array

        result = linprog(obj,
                         A_ub=all_vmat_neg,
                         b_ub=vmat_limit,
                         bounds=corr_bounds,
                         options=self.options,
                         method=self.method
                        )
        if result.success:
            print('Ordered State calculations completed...')
            if self.print_output:
                np.savetxt('ordered_correlations.out', result.x)
                with open('ordered_rho.out', 'w') as frho:
                    for vmat in self.cluster._vmat.values():
                        frho.write(f'{" ".join(map(str,vmat@result.x))}\n')
            return result
        print(
            f'WARNING: linear programming for ordered correlation search failed: {result.status} - {result.message}\nExiting...')
        return result

@dataclass
class ShortRangeOrderOptimization(ClusterOptimizer):

    options: dict
    disordered_correlations: np.ndarray
    ordered_correlations: np.ndarray
    structure: str
    num_trials: int = 100
    constr_tol: float = 1e-10
    early_stopping_count: int = 100
    norm_constrained: bool = True

    def __post_init__(self) -> None:

        self._mults_eci = self.cluster.clusmult_array * self.cluster.eci_array
        self._multconfig_kb = self.cluster.configmult_array * self.cluster.kb_array

        assert self.cluster.vmatrix_array.shape == (len(self._multconfig_kb), len(self._mults_eci))

        self._constraints = Constraints(self.cluster.vmatrix_array,
                                   self.ordered_correlations,
                                   self.disordered_correlations,
                                   self.norm_constrained
                                  ).constraints

        self._bounds = CorrBounds(self.cluster.num_clusters,
                                  len(self.cluster.single_point_clusters),
                                  self.disordered_correlations[self.cluster.single_point_clusters]
                                 ).sro_bounds

        self._structure_generator = get_random_structure(self.structure)

    def fit(self: ShortRangeOrderOptimization,
            temperature: float
           ) -> (float, np.ndarray, np.ndarray, float):

        earlystop = 0
        trial = 0

        result_correlation = self.disordered_correlations.copy()
        result_value = self._F(result_correlation,
                               self._mults_eci,
                               self._multconfig_kb,
                               self.cluster.vmatrix_array,
                               self._vrhologrho,
                               temperature
                              )

        result_constr_viol = np.float64(0.0)
        result_grad = np.zeros(result_correlation.shape[0])

        for trial in range(self.num_trials):

            accepted = False
            next(self._structure_generator)
            corrs_attempt = subprocess.run(['corrdump', '-c', f'-cf={self.structure}/{self.cluster._clusters_fname}', f'-s={self.structure}/randstr.in', f'-l={self.structure}/{self.cluster._lattice_fname}'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           check=True
                                          )
            # convert from bytes to string list
            corrs_attempt = corrs_attempt.stdout.decode('utf-8').split('\t')[:-1]
            corrs_attempt = np.array(corrs_attempt, dtype=np.float32)  # convert to arrays

            try:
                assert self.cluster.cluster_data.check_result_validity(corrs_attempt)
            except AssertionError:
                print('WARNING. Initial Structure Incorrect.')

            if trial == 0:
                corrs_attempt = self.disordered_correlations

            f_attempt = self._F(corrs_attempt,
                                self._mults_eci,
                                self._multconfig_kb,
                                self.cluster.vmatrix_array,
                                self._vrhologrho,
                                temperature
                               )

            temp_results = minimize(self._F,
                                    corrs_attempt,
                                    method='trust-constr',
                                    args=(
                                        self._mults_eci,
                                        self._multconfig_kb,
                                        self.cluster.vmatrix_array,
                                        self._vrhologrho,
                                        temperature,
                                    ),
                                    options=self.options,
                                    jac=self._dF,
                                    hess=self._d2F,
                                    constraints=self._constraints,
                                    bounds=self._bounds,
                                   )

            if temp_results is None:
                print('WARNING: Optimisation failed')

            elif temp_results.constr_violation < self.constr_tol and temp_results.fun < result_value:

                earlystop = 0
                result_value = temp_results.fun
                result_grad = temp_results.grad.copy()
                result_corr = temp_results.x.copy()
                result_constr_viol = temp_results.constr_violation

                accepted = True

            earlystop += 1

            if earlystop > self.early_stopping_count and trial > self.num_trials/2:
                print(
                    f'No improvement for consecutive {self.early_stopping_count} steps. After half of total steps ({int(self.num_trials/2)}) were done')
                break

            if self.print_output:
                print(f'Trial No.: {trial}')
                print(f'Current attempt correlations: {corrs_attempt}')
                print(f'Trial Validity: {self.cluster.check_result_validity(corrs_attempt)}')
                print(f'Attempt Free Energy @ T = {temperature}K : {f_attempt/self._num_lat_atoms}')
                print(f'Current Free Energy @ T = {temperature}K : {temp_results.fun/self._num_lat_atoms}')
                print(f'Current minimum correlations: {temp_results.x}')
                print(f"Gradient: {np.array2string(temp_results.grad)}")
                print(f"Constraint Violation: {temp_results.constr_violation}")
                print(
                    f"Stop Status: {temp_results.status} | {temp_results.message}")

                print(f"Acccepted? : {accepted}")
                print(f"Current Min Free Energy @ T = {temperature}K : {result_value/self._num_lat_atoms}")
                print(f"Current Best Contr. Viol : {result_constr_viol}")
                print('\n====================================\n')

        return (result_value, result_corr, result_grad, result_constr_viol)

#TODO
def ShortRangeOrderOptimization_Basinhopping(ShortRangeOrderOptimization):
    pass
