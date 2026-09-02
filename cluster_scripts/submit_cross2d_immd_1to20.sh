#!/bin/bash
# Submit IMMD runs 1–20 for 14 of 24 2D cross-polytope problems (batch 1 of 2).
# Batch 2 (remaining 10 problems) is in submit_cross2d_immd_batch2.sh — submit when queue < 450.
# Timing confirmed: ~2 min/iter (test job 11112958, BoothProblem_cross), 100 iters ≈ 3.75h.
# Partition: cpu (24h). Memory: 16G.

cd ~/repos/bosip_benchmarks

problems=(
    "RosenbrockProblem2_cross"
    "StyblinskiTangProblem2_cross"
    "MichalewiczProblem2_cross"
    "AckleyProblem2_cross"
    "AlpineProblem2_cross"
    "ExpandedSchafferF6Problem2_cross"
    "ExpandedZakharovProblem2_cross"
    "GriewankProblem2_cross"
    "RastriginProblem2_cross"
    "SalomonProblem2_cross"
    "SchwefelProblem2_cross"
    "SphereProblem2_cross"
    "BealeProblem_cross"
    "BoothProblem_cross"
)

job_ids=()
count=0
for pname in "${problems[@]}"; do
    for run_idx in $(seq 1 20); do
        job_name="${pname}_immd_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
