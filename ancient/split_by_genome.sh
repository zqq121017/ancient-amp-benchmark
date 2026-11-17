#!/bin/bash
#SBATCH -p general
#SBATCH -t 04:00:00
#SBATCH --mem=8g
#SBATCH -c 1
#SBATCH -o split.out
# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
cd $WORKDIR/modern_cohort
mkdir -p per_genome
# split by genome id = first two underscore fields of the header (HRGMv2_XXXX)
awk '/^>/{
       h=$1; sub(/^>/,"",h);
       n=split(h,a,"_"); g=a[1]"_"a[2];
       f="per_genome/" g ".faa"
     }
     { print >> f }' HRGMv2.sorfs
echo "genomes split:"; ls per_genome | wc -l
