#!/bin/bash
#SBATCH -p general
#SBATCH -t 08:00:00
#SBATCH --mem=16g
#SBATCH -c 1
#SBATCH -o getorf.out
# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
module load emboss/6.6.0
cd $WORKDIR/modern_cohort

# 1. concatenate all representative genomes into one FASTA
find HRGMv2_Rep_Genome -name "*.fna" -exec cat {} + > HRGMv2_all.fna

# 2. predict sORFs
getorf -sequence HRGMv2_all.fna -find 0 -table 11 -minsize 30 -maxsize 150 -outseq HRGMv2.sorfs
