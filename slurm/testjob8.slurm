#!/bin/bash

#SBATCH --mail-user=charles.johnson@accre.vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=400mb
#SBATCH --time=00:40:00
#SBATCH --output=testjob8.output
echo "I am tired...Zzzzzzzzz"
sleep 3600
echo "Waking up, Harrumph!"

