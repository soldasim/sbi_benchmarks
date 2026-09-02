#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=4G
#SBATCH --job-name=audit_group_b_eiv
#SBATCH --output=logs/audit_group_b_eiv_%j.out
#SBATCH --time=00:10:00

cd ~/repos/bosip_benchmarks
~/.juliaup/bin/julialauncher --project=src --eval 'include("src/audit_group_b_eiv_iters.jl")'
