#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=plot_proxy_cmp

export PATH="$HOME/.juliaup/bin:$PATH"

cd ~/repos/bosip_benchmarks
julia --project=src src/plot_proxy_comparison.jl
