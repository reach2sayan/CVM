import itertools
import numpy as np
from energyfunctions import F_efficient


def get_initial_trial(cluster_data,
                      corr_rnd,
                      T,
                      ord2disord_dist,
                      constraint,
                      trial_variance,
                      num_trials=100_000_00,
                      seed=42
                      ):

    rng = np.random.default_rng(seed)
    mult_arr = np.array(list(cluster_data.clustermult.values()))
    eci_arr = np.array(list(cluster_data.eci.values()))

    mults_eci = mult_arr * eci_arr

    all_vmat = np.vstack([vmat for vmat in cluster_data.vmat.values()])
    mults_config = np.array(
        list(itertools.chain.from_iterable(list(cluster_data.configmult.values()))))
    all_kb = np.array(list(itertools.chain.from_iterable([[kb for _ in range(
        len(cluster_data.configmult[idx]))] for idx, kb in cluster_data.kb.items()])))

    multconfig_kb = mults_config * all_kb
    assert all_vmat.shape == (len(multconfig_kb), len(mults_eci))
    vrhologrho = np.vectorize(lambda rho: rho * np.log(np.abs(rho)))

#    corr_trial = corr_rnd
#    fmin = F_efficient(corr_trial, mults_eci, multconfig_kb,
#                       all_vmat, vrhologrho, T)
#    corr_min = corr_rnd

    fixed_corrs = np.ones((num_trials,
                           len(cluster_data.single_point_clusters)+1
                           ))
    mean = np.zeros(cluster_data.num_clusters -
                    len(cluster_data.single_point_clusters) - 1)
    cov = trial_variance * \
        np.eye(cluster_data.num_clusters -
               len(cluster_data.single_point_clusters) - 1)
    random_corrs = np.random.multivariate_normal(mean, cov, num_trials)

    trial_corrs = np.hstack((fixed_corrs, random_corrs))
    trial_corrs = corr_rnd + trial_corrs
    trial_corrs = np.vstack((corr_rnd,trial_corrs))

    if constraint:
        def corr_validity(x): return (cluster_data.check_result_validity(x)) & (
            ord2disord_dist/2 - np.linalg.norm(x-corr_rnd) >= 0)
    else:
        def corr_validity(x): return (cluster_data.check_result_validity(x))

    valid_corrs = trial_corrs[np.apply_along_axis(corr_validity,
                                                  1,
                                                  trial_corrs
                                                  )
                              ]

    random_energy_eval = np.apply_along_axis(F_efficient,
                                             1,
                                             valid_corrs,
                                             mults_eci,
                                             multconfig_kb,
                                             all_vmat,
                                             vrhologrho,
                                             T
                                             )
    fmin, corr_min = np.amin(
        random_energy_eval), valid_corrs[np.argmin(random_energy_eval)]

#    for _ in range(num_trials):
#        ftrial = F_efficient(corr_trial, mults_eci,
#                             multconfig_kb, all_vmat, vrhologrho, T)
#
#        if (ftrial <= fmin) and (cluster_data.check_result_validity(corr_trial)):
#            if constraint:
#                if ord2disord_dist/2 - np.linalg.norm(corr_min-corr_rnd) >= 0:
#                    fmin = ftrial
#                    corr_min = corr_trial
#            else:
#                fmin = ftrial
#                corr_min = corr_trial
#
#        jitter = np.array([0,
#                           *[0] *
#                           len(cluster_data.single_point_clusters),
#                           *rng.normal(0,
#                                       trial_variance,
#                                       cluster_data.num_clusters -
#                                       len(cluster_data.single_point_clusters) - 1
#                                       )
#                           ]
#                          )
#        corr_trial = corr_rnd+jitter

    return fmin, corr_min
