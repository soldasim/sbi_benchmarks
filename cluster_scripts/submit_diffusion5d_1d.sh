#!/bin/bash
#SBATCH --job-name=diffusion5d_1d
#SBATCH --output=slurm-%A-%a.out
#SBATCH --array=1-5
#SBATCH --partition=cpufast
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

~/.juliaup/bin/julialauncher --project=src cluster_scripts/run_diffusion5d_1d.jl $SLURM_ARRAY_TASK_ID
