#!/bin/bash
#SBATCH --job-name=cdhit_pg
#SBATCH --output=cdhit_pg_%A_%a.out
#SBATCH --array=1-50
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=12:00:00

# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
module purge
module load cdhit
cd $WORKDIR/modern_cohort

mkdir -p clust50 clust80

# list all per-genome files, deterministically ordered
ls per_genome/*.faa | sort > genome_list.txt
TOTAL=$(wc -l < genome_list.txt)
NTASK=50
# this task handles every NTASK-th genome starting at its task id
i=${SLURM_ARRAY_TASK_ID}
while [ $i -le $TOTAL ]; do
    f=$(sed -n "${i}p" genome_list.txt)
    b=$(basename "$f" .faa)
    # 50% identity  (word size 3)
    cd-hit -i "$f" -o "clust50/${b}_nr50.faa" -c 0.5 -n 3 -l 5 -T 4 -M 6000 -d 0 > /dev/null 2>&1
    # 80% identity  (word size 5)
    cd-hit -i "$f" -o "clust80/${b}_nr80.faa" -c 0.8 -n 5 -l 5 -T 4 -M 6000 -d 0 > /dev/null 2>&1
    i=$(( i + NTASK ))
done
echo "task ${SLURM_ARRAY_TASK_ID} done"
