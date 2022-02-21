#!/bin/bash

#SBATCH --mail-user=sayan_samanta@brown.edu
#SBATCH --mail-type=ALL
#SBATCH -t 28:00:00
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128

source /ocean/projects/mat200006p/bhantai/CVM/bin/activate
module load openblas/0.3.12-gcc10.2.0 gcc/10.2.0

currdate=`date +"%b-%d"`
../../src/sro_correction --Tmin 100 --Tmax 2000 --Tstep 100 --no_constraint --log log-noconstraint-$currdate.out --global_trials 100 --out result-noconstraint-$currdate.json --
