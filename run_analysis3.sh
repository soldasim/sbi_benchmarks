#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module load Julia/1.10.0-linux-x86_64
cd ~/repos/bosip_benchmarks

# Try with system Julia first - check if JLD2 is available globally
julia -e 'import JLD2; println("JLD2 available")' 2>&1 | head -5
EOF

chmod +x run_analysis3.sh
sbatch run_analysis3.sh
