#!/bin/bash
#SBATCH --job-name=diffusion5d_test
#SBATCH --output=slurm-%j.out
#SBATCH --partition=cpufast
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

~/.juliaup/bin/julialauncher --project=src cluster_scripts/run_diffusion5d_precompute.jl 5 5
