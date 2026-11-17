#!/bin/bash
#SBATCH -p general
#SBATCH -t 08:00:00
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH -o ov_amps_1.out
# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
module purge; module load cdhit
cd $WORKDIR/modern_cohort
cd-hit-2d -i ancient1_consensus.faa -i2 $WORKDIR/AMPSphere_v.2021-03.faa -o ancient1_vs_AMPSphere.out -c 0.9 -n 5 -l 5 -T 8 -M 30000 -d 0
