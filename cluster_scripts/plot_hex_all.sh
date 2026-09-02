#!/bin/sh
#SBATCH --job-name=plot_hex_all
#SBATCH --partition=cpufast
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

export PATH="$HOME/.juliaup/bin:$PATH"
cd /home/soldasim/repos/bosip_benchmarks
julia --project=src src/plot_hex_all_results.jl && echo done > hex_all_plot.done
