#!/bin/bash

#SBATCH -t 12:00:00
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128

source /ocean/projects/mat200006p/bhantai/CVM/bin/activate
module load openblas/0.3.12-gcc10.2.0 gcc/10.2.0

currdate=`date +"%b-%d"`
./cvm_sro --phase=BCC_A2 --global_trials=50 --out=noch-$currdate.json --logfile=log_noch-$currdate.out


