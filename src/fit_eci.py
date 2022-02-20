import subprocess
import numpy as np
from scipy.linalg import lstsq, LinAlgError
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score

def fit_eci_scipy(clusters_fit, correlations, energies, structure):
    """
    Module to fit ECIs to a set of energies and corresponding correlations:
    Input: clusters_fit - cluster description for the fitting.
           correlations - dictionary containing all the correlations, index by the folder name
           energies     - corresponding energies also indexed by folder name
    Output: Fitted ECIs.
            Note, this gets padded with zeros to match the number of clusters of the full cluster description.
    """

    corrs, energies_array = map(np.array, zip(
        *[(correlations[key], energies[key]) for key in energies.keys()]))
    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
    mults_corrs = corrs*mults
    np.savetxt(f'{structure}/ref_correlations.in', mults_corrs)
    np.savetxt(f'{structure}/ref_energies.in', energies_array)
    try:
        eci_fit, _, _, _ = lstsq(
            mults_corrs, energies_array, lapack_driver='gelsd')
    except LinAlgError as ae:
        print(f'Issues with fitting ECI: {ae}')

    return eci_fit


def fit_eci_lsfit(clusters_fit, correlations, energies, structure):
    """
    Module to fit ECIs to a set of energies and corresponding correlations:
    Input:
        clusters_fit - cluster description for the fitting.
        correlations - dictionary containing all the correlations, index by the folder name
        energies     - corresponding energies also indexed by folder name
    Output:
        Fitted ECIs.
        Note, this gets padded with zeros to match the number of clusters of the full cluster description but not here
    """

    corrs, energies_array = map(np.array, zip(
        *[(correlations[key], energies[key]) for key in energies.keys()]))
    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
    mults_corrs = corrs*mults
    np.savetxt(f'{structure}/ref_correlations.in', mults_corrs)
    np.savetxt(f'{structure}/ref_energies.in', energies_array)

    eci_fit = subprocess.run(['lsfit', f'-x={structure}/ref_correlations.in', f'-y={structure}/ref_energies.in', '-colin'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             check=True
                             )
    eci_fit = np.fromstring(eci_fit.stdout.decode('utf-8'), sep='\n')

    return eci_fit

def fit_eci_ransac(clusters_fit, correlations, energies, structure):

    corrs, energies_array = map(np.array, zip(
        *[(correlations[key], energies[key]) for key in energies.keys()]))
    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
    mults_corrs = corrs*mults
    np.savetxt(f'{structure}/ref_correlations.in', mults_corrs)
    np.savetxt(f'{structure}/ref_energies.in', energies_array)

    lr = LinearRegression(fit_intercept = False)
    ransac = RANSACRegressor(base_estimator=lr,loss='squared_loss')
    ransac.fit(mults_corrs, energies_array)
    for idx, outlier in enumerate(np.logical_not(ransac.inlier_mask_)):
        if outlier:
            print(f'Outlier correlation: {mults_corrs[idx]}')
            print(f'Outlier Energy: {energies_array[idx]}')
    print(f'Mean Square Error: {mean_squared_error(energies_array[ransac.inlier_mask_], mults_corrs[ransac.inlier_mask_] @ ransac.estimator_.coef_ )}')
    print(f'R2 score: {r2_score(energies_array[ransac.inlier_mask_], mults_corrs[ransac.inlier_mask_] @ ransac.estimator_.coef_ )}')
    return ransac.estimator_.coef_



## Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
#
## * Redistributions of source code must retain the above copyright
## notice, this list of conditions and the following disclaimer.
#
## * Redistributions in binary form must reproduce the above
## copyright notice, this list of conditions and the following
## disclaimer in the documentation and/or other materials provided
## with the distribution.
#
## * Neither the name of the Andrew D. Straw nor the names of its
## contributors may be used to endorse or promote products derived
## from this software without specific prior written permission.
#
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
#def fit_eci_ransac(clusters_fit, correlations, energies,
#                   seed, min_points, max_iterations, threshold,
#                   min_inliners,
#                   ):
#    """fit model parameters to data using the RANSAC algorithm
#    This implementation written from pseudocode found at
#    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
#    Input:
#        clusters_fit - cluster description for the fitting.
#        correlations - dictionary containing all the correlations, index by the folder name
#        energies     - corresponding energies also indexed by folder name
#        seed         - random number seed for reprodubicle experiments
#        basemodel    - a model that can be fitted to data points
#        min_points   - the minimum number of data values required to fit the model
#        max_iterations - the maximum number of iterations allowed in the algorithm
#        threshold    - a threshold value for determining when a data point fits a model
#        min_inliners - the number of close data values required to assert that a model fits well to data
#    Output:
#        Fitted ECIs.
#        Note, this gets padded with zeros to match the number of clusters of the full cluster description but not here
#    """
#
#    corrs, energies_array = map(np.array, zip(
#        *[(correlations[key], energies[key]) for key in energies.keys()]))
#    mults = np.array([clus['mult'] for clus in clusters_fit.clusters.values()])
#    mults_corrs = corrs*mults
#    data = np.c_[mults_corrs, energies_array]
#
#    ransac_basemodel = LinearLeastSquaresModel()
#    ransac_regressor = RansacRegressor(base_model=ransac_basemodel,
#                                       min_points=min_points,
#                                       max_iterations=max_iterations,
#                                       threshold=threshold,
#                                       min_inliners=min_inliners,
#                                       )
#    print(vars(ransac_regressor))
#    eci_fit, inliers = ransac_regressor.fit(data, seed)
#
#    return eci_fit, inliers
