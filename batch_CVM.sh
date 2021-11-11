#!/bin/bash

#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4g

module load python/3.7.4
source /users/ssamanta/scratch/CVM/bin/activate

./cvm_pd --Tmax 1 --Tmin 1 --nTemp 1 --out test.json --maxiter 1000 --global_trials 50 --logfile test.log



