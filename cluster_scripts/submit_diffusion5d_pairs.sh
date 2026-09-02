#!/bin/bash
#SBATCH --job-name=diffusion5d_pair
#SBATCH --output=slurm-%A-%a.out
#SBATCH --array=1-10
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

~/.juliaup/bin/julialauncher --project=src cluster_scripts/run_diffusion5d_pair.jl $SLURM_ARRAY_TASK_ID
