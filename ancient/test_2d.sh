#!/bin/bash
#SBATCH --job-name=cdhit2d_test
#SBATCH -p general
#SBATCH -t 1-00:00:00
#SBATCH --mem=64g
#SBATCH -c 8
#SBATCH -o cdhit2d_test.out
# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
module purge
module load cdhit
cd $WORKDIR/modern_cohort
# db1 = ancient (small, in memory); db2 = modern (141M, streamed)
# 90% identity: word size -n 5
cd-hit-2d -i ancient1.faa -i2 HRGMv2_modern_nr50.faa -o ancient1_vs_modern.out \
          -c 0.9 -n 5 -l 5 -T 8 -M 60000 -d 0
