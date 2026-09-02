#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=classify_posteriors
#SBATCH --output=logs/classify_posteriors_%j.out
#SBATCH --time=00:30:00

cd ~/repos/bosip_benchmarks
JULIA_NUM_THREADS=4 ~/.juliaup/bin/julialauncher --project=src --eval 'include("src/main.jl"); include("src/classify_posteriors.jl")'
