#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=setup_sharp

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=4

cd ~/repos/bosip_benchmarks
julia --project=src src/setup_sharp_opt_problems.jl
