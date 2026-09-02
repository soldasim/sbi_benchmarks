#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p cpufast

export JULIA_NUM_THREADS=4
cd ~/repos/bosip_benchmarks
~/.juliaup/bin/julialauncher --project=src src/run_classify_posteriors.jl
