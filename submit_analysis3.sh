#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -p cpu

cd ~/repos/bosip_benchmarks
~/.juliaup/bin/julia --project=src analyze_bosip_fixed.jl
