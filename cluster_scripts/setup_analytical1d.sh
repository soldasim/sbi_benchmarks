#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=setup_analytical1d

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=8

cd ~/repos/bosip_benchmarks
julia --project=src src/setup_analytical1d.jl
