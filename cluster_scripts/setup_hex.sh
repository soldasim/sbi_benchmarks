#!/bin/sh
#SBATCH --job-name=setup_hex
#SBATCH --partition=cpufast
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

export PATH="$HOME/.juliaup/bin:$PATH"
cd /home/soldasim/repos/bosip_benchmarks
julia --project=src src/setup_hex_opt_problems.jl
