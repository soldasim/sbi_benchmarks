#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p cpufast

cd ~/repos/bosip_benchmarks
~/.juliaup/bin/julialauncher --project=src src/compute_acq_scores.jl
