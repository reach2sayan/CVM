from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from functools import cached_property
import itertools
import numpy as np
from atatio import read_clusters, read_kbcoeffs, read_configmult, read_clustermult, read_eci, read_configs, read_vmatrix

EPSILON = 1e-2

@dataclass(frozen=True)
class Cluster:
    """
    Class to hold Cluster Description
    """

    _clusters_fname: InitVar[str] = field(default='clusters.out', repr=False)
    _eci_fname: InitVar[str] = field(default='eci.out', repr=False)
    _clustermult_fname: InitVar[str] = field(default='clusmult.out', repr=False)
    _config_fname: InitVar[str] = field(default='config.out', repr=False)
    _configmult_fname: InitVar[str] = field(default='configmult.out', repr=False)
    _kb_fname: InitVar[str] = field(default='configkb.out', repr=False)
    _vmat_fname: InitVar[str] = field(default='vmat.out', repr=False)
    _lattice_fname: str = field(default='lat.in',repr=False)

    _clusters: dict = field(init=False)
    _kb: dict = field(init=False)
    _configmult: dict = field(init=False)
    _clustermult: dict = field(init=False)
    _configs: dict = field(init=False)
    _vmat: dict = field(init=False)
    _eci: dict = field(init=False)

    def __post_init__(self: Cluster,
                      _clusters_fname: str,
                      _eci_fname: str,
                      _clustermult_fname: str,
                      _config_fname: str,
                      _configmult_fname: str,
                      _kb_fname: str,
                      _vmat_fname: str,
                     ) -> None:

        object.__setattr__(self, '_clusters', read_clusters(_clusters_fname))
        object.__setattr__(self, '_kb', read_kbcoeffs(_kb_fname))
        object.__setattr__(self, '_clustermult', read_clustermult(_clustermult_fname))
        object.__setattr__(self, '_configmult', read_configmult(_configmult_fname))
        object.__setattr__(self, '_configs', read_configs(_config_fname))
        object.__setattr__(self, '_vmat', read_vmatrix(_vmat_fname))
        object.__setattr__(self, '_eci', read_eci(_eci_fname))

    @cached_property
    def num_configs(self: Cluster) -> int:
        return len(self._configmult)

    @cached_property
    def single_point_clusters(self: Cluster) -> list:
        return [cluster_idx for cluster_idx, cluster in self._clusters.items() if cluster['type'] == 1]

    @cached_property
    def num_clusters(self: Cluster) -> int:
        return len(self._clusters)

    @cached_property
    def clusmult_array(self: Cluster) -> np.ndarray:
        return np.array(list(self._clustermult.values()))

    @cached_property
    def eci_array(self: Cluster) -> np.ndarray:
        return np.array(list(self._eci.values()))

    @cached_property
    def configmult_array(self: Cluster) -> np.ndarray:
        return np.array(list(itertools.chain.from_iterable(list(self._configmult.values()))))

    @cached_property
    def kb_array(self: Cluster) -> np.ndarray:
        return np.array(list(itertools.chain.from_iterable([[kb for _ in range(
            len(self._configmult[idx]))] for idx, kb in self._kb.items()])))

    @cached_property
    def vmatrix_array(self: Cluster) -> np.ndarray:
        return np.vstack(list(self._vmat.values()))
