#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=plot_posteriors_sharp

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=4

cd ~/repos/bosip_benchmarks
julia --project=src src/plot_all_opt_posteriors_sharp.jl
