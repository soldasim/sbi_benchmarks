#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

. /etc/profile.d/modules.sh
module load Julia/1.10.0-linux-x86_64
cd ~/repos/bosip_benchmarks
which julia
julia -e 'import JLD2; println(JLD2 available)'
