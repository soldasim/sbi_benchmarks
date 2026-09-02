#!/bin/bash
# Submit EIV runs for the 12 5D cross-polytope opt-function problems (5 runs each).
# Run from ~/repos/bosip_benchmarks after setup_cross5d_problems.jl has completed.

cd ~/repos/bosip_benchmarks

problems=(
    "RosenbrockProblem5_cross"
    "StyblinskiTangProblem5_cross"
    "MichalewiczProblem5_cross"
    "AckleyProblem5_cross"
    "AlpineProblem5_cross"
    "ExpandedSchafferF6Problem5_cross"
    "ExpandedZakharovProblem5_cross"
    "GriewankProblem5_cross"
    "RastriginProblem5_cross"
    "SalomonProblem5_cross"
    "SchwefelProblem5_cross"
    "SphereProblem5_cross"
)

job_ids=()
count=0
for pname in "${problems[@]}"; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
