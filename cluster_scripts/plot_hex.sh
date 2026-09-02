#!/bin/sh
#SBATCH --job-name=plot_hex
#SBATCH --partition=cpufast
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

export PATH="$HOME/.juliaup/bin:$PATH"
cd /home/soldasim/repos/bosip_benchmarks
julia --project=src src/plot_hex_results.jl && echo done > hex_plot.done
