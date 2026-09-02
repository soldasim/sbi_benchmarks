#!/bin/sh
#SBATCH --job-name=setup_hex_all
#SBATCH --partition=cpufast
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

export PATH="$HOME/.juliaup/bin:$PATH"
cd /home/soldasim/repos/bosip_benchmarks
julia --project=src src/setup_hex_all_problems.jl && echo done > hex_all_setup.done
