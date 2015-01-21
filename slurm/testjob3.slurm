 #!/bin/bash

#SBATCH --mail-user=your_email@accre.vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=400mb
#SBATCH --time=00:40:00
#SBATCH --output=testjob3.output
./test3

