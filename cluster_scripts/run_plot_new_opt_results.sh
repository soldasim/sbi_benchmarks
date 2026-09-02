#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=plot_new_opt

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=4

cd ~/repos/bosip_benchmarks
julia --project=src src/plot_new_opt_results.jl
