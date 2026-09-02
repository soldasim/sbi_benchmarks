#!/bin/bash
# Continue EIV runs for the 12 5D cross-polytope opt-function problems (5 runs each).
# All 60 original jobs (submit_cross5d_eiv.sh) timed out at 24h (~50/200 iters done).
# This script continues from checkpoints with a 2-day wall time.
# Run from ~/repos/bosip_benchmarks.

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
        job_name="${pname}_eiv_${run_idx}_cont"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=16G --time=2-00:00:00 --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 1 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
