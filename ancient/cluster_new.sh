#!/bin/bash
#SBATCH --job-name=cdhit_new
#SBATCH --output=cdhit_%A_%a.out
#SBATCH --array=1-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=1-00:00:00

# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
module purge
module load cdhit

cd $WORKDIR/new_ancient
D=data${SLURM_ARRAY_TASK_ID}
cd-hit -i ${D}.fasta -o ${D}_nr50.fasta -c 0.5 -n 3 -l 5 -T 8 -M 30000 -d 0
echo "done $D"