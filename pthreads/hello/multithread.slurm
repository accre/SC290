#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 # 1 task/process will spawn 4 threads
#SBATCH --cpus-per-task=4  # 1 cpu core per thread
#SBATCH --time=10   # ten minutes
#SBATCH --mem-per-cpu=1G
#SBATCH --output=multithread.out
#SBATCH --mail-user=vunetid@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=multithreaded_example

# Run multi-threaded application
./hello