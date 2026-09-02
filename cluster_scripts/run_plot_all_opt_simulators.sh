#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=plot_simulators

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=2

cd ~/repos/bosip_benchmarks
julia --project=src src/plot_all_opt_simulators.jl
