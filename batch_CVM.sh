#!/bin/bash

#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4g
#SBATCH --mail-user=sayan_samanta@brown.edu
#SBATCH --mail-type=ALL

module load python/3.7.4
source /users/ssamanta/scratch/CVM/bin/activate

./cvm_pd --Tmax 1000 --Tmin 1000 --nTemp 1 --out test.csv



