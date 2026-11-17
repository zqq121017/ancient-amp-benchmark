#!/bin/bash
#SBATCH -p general
#SBATCH -t 04:00:00
#SBATCH --mem=4g
#SBATCH -c 1
#SBATCH -o dl.out
# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
cd $WORKDIR/modern_cohort
wget -c https://www.decodebiome.org/HRGM/data/genome_catalog/HRGMv2_Genomes/HRGMv2_Rep_Genome.tar.gz
