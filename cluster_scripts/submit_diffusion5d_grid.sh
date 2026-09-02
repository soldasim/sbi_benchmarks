#!/bin/bash
#SBATCH --job-name=diffusion5d_grid
#SBATCH --output=slurm-%j.out
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

~/.juliaup/bin/julialauncher --project=src cluster_scripts/run_diffusion5d_precompute.jl
