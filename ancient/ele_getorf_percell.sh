#!/bin/bash
#SBATCH --job-name=ele_getorf
#SBATCH -p general
#SBATCH -t 08:00:00
#SBATCH --mem=8g
#SBATCH -c 1
#SBATCH -o ele_getorf_%a.out
#SBATCH --array=1-20
# Set WORKDIR to your project directory before running, or edit the default below.
: "${WORKDIR:=.}"
module load emboss/6.6.0
cd $WORKDIR/mammoth_cohort

TOTAL=$(wc -l < ele_genome_list.txt)
NTASK=20
i=${SLURM_ARRAY_TASK_ID}
while [ $i -le $TOTAL ]; do
    f=$(sed -n "${i}p" ele_genome_list.txt)
    b=$(basename "$f" .fa.gz)        # e.g. GCMeta_00340076
    zcat "$f" > tmp_${b}.fna
    getorf -sequence tmp_${b}.fna -find 0 -table 11 -minsize 30 -maxsize 150 \
           -outseq per_genome_sorfs/${b}.sorfs > /dev/null 2>&1
    rm -f tmp_${b}.fna
    i=$(( i + NTASK ))
done
echo "task ${SLURM_ARRAY_TASK_ID} done"
