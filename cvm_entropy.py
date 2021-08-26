#!/usr/bin/env python3

import numpy as np
import argparse
from data_reader import *
from pprint import pprint as pp
import os
import math
from scipy.optimize import minimize
from scipy.optimize import SR1, BFGS
from scipy.optimize import Bounds
from ase.units import kB

def read_input_fname(args):

    clusters = read_clusters(args.clusters)
    eci = read_eci(args.eci)
    kb = read_kb_coefficients(args.kikuchi_baker)
    vmat = read_vmat(args.vmat)
    configs = read_configs(args.configs)
    configcoef = read_configcoeff(args.configcoef)

    return clusters, eci, kb, vmat, configcoef, configs

def constraint_rhos_sum(corrs, vmat, clusters, configcoef,):

    rho_sum = []

    def clus_prob(cluster_idx):
        rho = np.matmul(vmat[cluster_idx],corrs)
        return rho

    for cluster_idx, _ in clusters.items():
        rho = clus_prob(cluster_idx)
        rho_sum.append(np.sum(configcoef[cluster_idx]*rho))

    return np.sum(1 - np.array(rho_sum))

def constraint_rhos_single(corrs, vmat, clusters, configcoef,):

    all_pos = []

    def clus_prob(cluster_idx):
        rho = np.matmul(vmat[cluster_idx],corrs)
        return rho

    for cluster_idx, _ in clusters.items():
        rho = clus_prob(cluster_idx)
        all_pos.append(((rho >= 0).all() and (rho <= 1).all()))

    return 1 - int(all(all_pos))

def constraint_singlet(corrs):

    return corrs[1] - FIXED_CORR_1

def constraint_zero(corrs):

    return corrs[0] - 1.0

def F(corrs, vmat, kb, clusters, configs, configcoef,):

    S = 0
    H = 0
    #corrs = np.concatenate((corrs_fixed,corrs_optim))

    def clus_prob(cluster_idx):
        rho = np.matmul(vmat[cluster_idx],corrs)
        print(cluster_idx)
        print(vmat)
        print(corrs)
        return rho

    def inner_sum(cluster_idx):
        isum = 0
        rho = clus_prob(cluster_idx)
#        print(rho)
        try:
            for i in range(configs[cluster_idx]['num_of_subclus']):
                isum += configcoef[cluster_idx][i] * rho[i] * math.log(rho[i])
        except ValueError as ve:
            #print(ve)
            pass
        #for i in range(configs[cluster_idx]['num_of_subclus']):
        #        isum += configcoef[cluster_idx][i] * rho[i] * np.log(rho[i])
        return isum

    for cluster_idx, cluster in clusters.items():
        H += cluster['mult']*eci[cluster_idx]*corrs[cluster_idx]
        S += kb[cluster_idx]*inner_sum(cluster_idx)

    T = 3000

    #print(corrs)
    #return S
    return H - T*kB*S


cwd = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument('--eci','-e',
                    default=f'{cwd}/eci.out', 
                    help="filename containing ECI's")

parser.add_argument('--clusters','-c',
                    default=f'{cwd}/clusters.out', 
                    help="filename containing maximal cluster information")

parser.add_argument('--configs','-cfg',
                    default=f'{cwd}/config.out', 
                    help="filename containing maximal cluster information")

parser.add_argument('--kikuchi_baker','-kb',
                    default=f'{cwd}/kb.out', 
                    help="filename containing the Kikuchi Baker Coefficients")

parser.add_argument('--maxclus','-maxc',
                    default=f'{cwd}/maxclus.in', 
                    help="filename containing the Maximal Cluster Interaction")

parser.add_argument('--lattice','-l',
                    default=f'{cwd}/lat.in', 
                    help="filename containing the Underlying lattice")

parser.add_argument('--vmat','-v',
                    default=f'{cwd}/vmat.out', 
                    help="filename containing the VMatrix for each maximal cluster")

parser.add_argument('--configcoef','-cc',
                    default=f'{cwd}/configcoef.out', 
                    help="filename containing the coefficient for each subcluster")

args = parser.parse_args()

clusters, eci, kb, vmat, configcoef, configs = read_input_fname(args)

FIXED_CORR_1 = 0.0
corrs = np.array([1.0,FIXED_CORR_1,0.02041,0.02041,0.00292,0.0])

bounds_corrs = Bounds(np.array([-1]*len(corrs)),np.array([1]*len(corrs)))

#print('CLusters')
#pp(clusters)
#print('ECI')
#pp(eci)
#print('kb')
#pp(kb)
#print('vmat')
#pp(vmat)
#print('configcoef')
#pp(configcoef)
#print('clusters')
#pp(configs)

res = minimize(F,
               corrs,
               args=(vmat, kb, clusters, configs, configcoef,),
               method='trust-constr',
               options={'verbose': 1,},#'gtol': 1e-3, 'maxiter': 200},
               jac='3-point', #hess=SR1(),
               constraints=[{'fun': constraint_rhos_sum, 'type': 'eq', 'args': (vmat, clusters, configcoef,)},
                            {'fun': constraint_rhos_single, 'type': 'eq', 'args': (vmat, clusters, configcoef,)},
                            {'fun': constraint_singlet, 'type': 'eq'},
                            {'fun': constraint_zero, 'type':'eq'},
                           ],
               bounds=bounds_corrs
              )

for cluster_idx in clusters:
    print(np.matmul(vmat[cluster_idx],))

print('=====================')
print(res.x)
