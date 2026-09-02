#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH -p cpufast

export JULIA_NUM_THREADS=24
cd ~/repos/bosip_benchmarks
~/.juliaup/bin/julialauncher --project=src src/precompute_marginals_diffusion.jl
