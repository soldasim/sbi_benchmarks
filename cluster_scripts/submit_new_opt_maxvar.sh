#!/bin/bash
# Submit MaxVar runs for all 21 new opt-function problems (2D, 5 runs each).
# Run from ~/repos/bosip_benchmarks after setup_new_opt_problems.sh has completed.

cd ~/repos/bosip_benchmarks

problems=(
    # d-dimensional at x_dim=2
    "AckleyProblem2"
    "AlpineProblem2"
    "ExpandedSchafferF6Problem2"
    "ExpandedZakharovProblem2"
    "GriewankProblem2"
    "RastriginProblem2"
    "SalomonProblem2"
    "SchwefelProblem2"
    "SphereProblem2"
    # 2D-only
    "BealeProblem"
    "BealeProxyProblem"
    "BoothProblem"
    "CrossInTrayProblem"
    "DropWaveProblem"
    "EasomProblem"
    "GoldsteinPriceProblem"
    "GoldsteinPriceProxyProblem"
    "HimmelblauProblem"
    "HolderTableProblem"
    "LeviN13Problem"
    "MatyasProblem"
    "SchafferN2Problem"
    "ThreeHumpCamelProblem"
)

count=0
for pname in "${problems[@]}"; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" maxvar "$run_idx" 0 200 nothing
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
