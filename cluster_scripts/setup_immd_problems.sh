#!/bin/bash
#SBATCH -p cpulong
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --job-name=setup-immd
#SBATCH --output=slurm-%j.out

cd ~/repos/bosip_benchmarks
~/.juliaup/bin/julialauncher --project=src -e 'include("src/setup_immd_problems.jl")'
