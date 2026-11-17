# Benchmarking Machine Learning Methods for Antimicrobial Peptide Prediction and Discovery in Ancient Microbiomes
<img width="2322" height="3159" alt="fig1_v13-1" src="https://github.com/user-attachments/assets/7a7af397-2c2b-4657-a35e-fc70de268b98" />

 
**Overview**  
This repository accompanies *Benchmarking Machine Learning Methods for Antimicrobial Peptide Prediction and Discovery in Ancient Microbiomes* (Preprint available on bioRxiv). Antimicrobial peptides (AMPs) are a promising alternative to conventional antibiotics, yet it remains unclear which feature representations and prediction methods best distinguish AMPs from non-antimicrobial sequences, and ancient microbiomes have received little attention as a source of novel candidates. We benchmark six machine learning methods across twelve feature configurations and eight curated AMP datasets under a unified framework, then apply the best-performing approach to three ancient metagenomes (Neanderthal gut, ancient oral, and mammoth-associated communities) to nominate high-confidence candidate AMPs. We also use reinforcement learning as a complementary, label-free route to candidate discovery.
 
**Key Features**  
- Unified Benchmark: Six methods (RF, GCN, APIN, EBiLSTM, TabICL, RealMLP) compared across twelve feature configurations and eight datasets on common ground  
- Feature Representations: One-hot, position encoding, amino acid composition, physicochemical properties, and ESM-2 and ProtBERT language model embeddings, used individually and in combination  
- Ancient Microbiome Screening: Application of the benchmarked pipeline to three ancient metagenomes to nominate candidate AMPs  
- Body-Site-Matched Overlap: Sequence comparison of ancient candidate sORFs against modern references matched by body site and host  
- Complementary Selection: Reinforcement-learning search that prioritizes candidates from the ancient peptide pools using physicochemical and diversity criteria, without modern AMP labels
**Repository Structure**  
```
benchmark/     Benchmark of six ML methods across twelve feature configurations
               (six 10fold_run_*.py scripts and the shared data_processing.py)
ancient/       Ancient-microbiome application: predict_ancient.py and the
               sORF-calling, clustering, and sequence-overlap pipeline (shell scripts)
rl/            Reinforcement-learning candidate selection: PPO and GFlowNet
               searches over the ancient peptide pools (src/ and configs/)
```
 
**Tested Environment**  
Developed and tested with:  
- Python: 3.12.4
- PyTorch: 2.7.1+cu118 (CUDA 11.8) 
- PyTorch Geometric: 2.6.1 
- NumPy: 2.1.2  
- Pandas: 2.3.3 
- Scikit-learn: 1.6.1 
- SciPy: 1.16.2
External tools for the ancient pipeline: EMBOSS 6.6.0 (getorf) and CD-HIT (cd-hit and cd-hit-2d). The TabICL method requires the `tabicl` package. The reinforcement-learning component runs on Python 3.10 to 3.12 with its own `rl/requirements.txt`.

**Usage**
 
**Benchmark**  
Each method is run with a single script; all six import the shared `data_processing.py`. For example:
```
cd benchmark
python 10fold_run_apin_12_feature_all.py
python 10fold_run_tabicl_12_feature_all.py
```
Each script evaluates its method across the twelve feature configurations using ten-fold cross-validation and writes per-configuration performance to its results directory.
 
**Ancient Prediction**  
`predict_ancient.py` trains one method on the combined training set (ESM-2 features, configuration 3) and scores the ancient datasets. Paths are passed on the command line:
```
cd ancient
python predict_ancient.py --method apin \
    --train-pkl data/combined_training.pkl \
    --ancient-dir data/ancient \
    --out-dir predictions
```
Run once per method (`apin`, `ebilstm`, `tabicl`); a peptide is nominated as a high-confidence candidate when all three methods assign a probability of at least 0.9999.
 
**Ancient sORF and Overlap Pipeline**  
The shell scripts in `ancient/` call sORFs from genome catalogues and compare the ancient sORFs against modern references. Set `WORKDIR` to your project directory (the scripts default to the current directory) and submit them in order:
```
sbatch get_HRGMv2.sh            # download a modern reference catalogue
sbatch run_getorf.sh            # predict sORFs from the genomes (EMBOSS getorf)
sbatch split_by_genome.sh       # split sORFs into per-genome files
sbatch cluster_per_genome.sh    # per-genome CD-HIT clustering (50% identity)
sbatch test_2d.sh               # cd-hit-2d overlap: ancient vs modern (90% identity)
sbatch cluster_new.sh           # cluster the ancient sORF sets (50% identity)
sbatch ov_amps_1.sh             # overlap of ancient predicted AMPs vs AMPSphere
```
The scripts are written for a Slurm cluster; adjust the `#SBATCH` directives, module names, and resources for your environment.
 
The modern-reference scripts above are written for the human gut catalogue (HRGM2). The ancient oral and mammoth-associated analyses use the same pipeline with their matched references (HROM for oral; elephant gut MAGs for mammoth), changing only the input paths and the genome-ID field used when splitting. The elephant catalogue stores each genome as a separate gzipped file without a shared genome identifier in the sequence headers, so its sORFs are called per genome with `ele_getorf_percell.sh` in place of the concatenate-then-split step. Key parameters are identical across cohorts: getorf (`-find 0 -table 11 -minsize 30 -maxsize 150`), per-genome CD-HIT at 50% identity (`-c 0.5 -n 3`), and cd-hit-2d at 90% identity (`-c 0.9 -n 5`).
 
**Reinforcement-Learning Candidate Selection**  
As a complementary, label-free approach, PPO and GFlowNet search the ancient peptide pools under two reward designs: S1, based on amino-acid composition entropy, and S2, based on the Porto et al. physicochemical fitness function. No AMP prediction score is used during search or ranking. The rewards combine mean quality, pairwise diversity, cluster coverage, and a redundancy penalty as `r = (Q + D + C + 1 - U) / 4`. Inputs are the C3 (ESM-2) features, mean-pooled over residues and zero-padded from 40 to 64 dimensions, with no additional PCA fitted; the prepared feature and reward caches are supplied separately. Search runs on CPU with Python 3.10 to 3.12.
```
cd rl
pip install -r requirements.txt
 
# prepare features for one dataset (gut, oral, or mammoth)
python src/features.py --dataset gut --conversion-dir <c3_dir> \
    --cache-root <cache_dir> --output <actor_dir>
 
# validate, then run one configuration
python src/search.py --action validate --dataset gut --inputs <actor_dir>/inputs.json
python src/search.py --action run --dataset gut --inputs <actor_dir>/inputs.json \
    --output <run_dir> --route S1 --method ppo --seed 17
```
Repeat the search over both rewards (S1, S2), both methods (`ppo` and `tb` for GFlowNet), the three seeds (17, 29, 43), and all three datasets, then aggregate the candidate lists:
```
python src/outputs.py --runs-root <runs_root> --output <candidates_dir>
```
Each run ranks the distinct peptides encountered during search by their quality score and retains the top candidates per dataset (111 gut, 76 oral, 50 mammoth), producing one `sequence` column per list. Configuration lives in `configs/experiment.json`; see `python src/convert_c3.py --help` for converting the C3 feature archive.
 
**Data Availability**  
Ancient metagenomes (European Nucleotide Archive): Neanderthal gut, PRJEB41665; ancient oral (dental calculus), PRJEB43389; mammoth-associated, PRJEB78615.  
Modern references: Human Reference Gut Microbiome v2 (HRGM2), Human Reference Oral Microbiome (HROM), elephant gut metagenome-assembled genomes (gcMeta), and AMPSphere.
