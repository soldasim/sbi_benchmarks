#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -p cpu
#SBATCH -o ~/repos/bosip_benchmarks/plot_job_%j.out
#SBATCH -e ~/repos/bosip_benchmarks/plot_job_%j.err

cd ~/repos/bosip_benchmarks
~/.juliaup/bin/julia --project=src src/plot_sir_duffing_comparison.jl
