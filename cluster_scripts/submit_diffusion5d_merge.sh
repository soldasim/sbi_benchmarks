#!/bin/bash
#SBATCH --job-name=diffusion5d_merge
#SBATCH --output=slurm-%j.out
#SBATCH --partition=cpufast
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

~/.juliaup/bin/julialauncher --project=src cluster_scripts/run_diffusion5d_merge.jl
