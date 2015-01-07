#!/bin/bash
#
# To run type: bash make_dirs.sh
#

# loop incrementing by 1 from i=1 to i=20,
for i in `seq 1 20`
do

  mkdir run$i # create new directory 
  cd run$i # move into new directory
  echo "Hello from run"$i > foo.txt # create a new file foo.txt with Hello line
  cd ../ # move back into parent directory

done