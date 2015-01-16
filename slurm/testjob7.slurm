#!/bin/bash

#SBATCH --mail-user=charles.johnson@accre.vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=400mb
#SBATCH --time=00:45:00
#SBATCH --output=testjob7.output
echo "I am tired...Zzzzzzzzz"
sleep 2400
echo "Waking up, Harrumph!"

