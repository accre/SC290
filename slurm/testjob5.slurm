#!/bin/bash

#SBATCH --mail-user=your_name@accre.vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=400mb
#SBATCH --time=00:15:00
#SBATCH --output=testjob5.output
echo "This is not good"
touch /usr/local/perl/5.16.2/x86_64/sysgcc/nonet/source/Configure
