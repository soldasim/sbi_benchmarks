#!/bin/bash
# Submit IMMD runs 1–20 for the remaining 10 2D cross-polytope problems (batch 2 of 2).
# Batch 1 (submit_cross2d_immd_1to20.sh, problems 1–14) was submitted first.
# Submit when queue total < 450 (need 200 slots free).
#
# IMMD at d=2: ~2 min/iter, 100 iters ≈ 3.75h. Partition: cpu (24h).

cd ~/repos/bosip_benchmarks

problems=(
    "CrossInTrayProblem_cross"
    "DropWaveProblem_cross"
    "EasomProblem_cross"
    "GoldsteinPriceProblem_cross"
    "HimmelblauProblem_cross"
    "HolderTableProblem_cross"
    "LeviN13Problem_cross"
    "MatyasProblem_cross"
    "SchafferN2Problem_cross"
    "ThreeHumpCamelProblem_cross"
)

job_ids=()
count=0
for pname in "${problems[@]}"; do
    for run_idx in $(seq 1 20); do
        job_name="${pname}_immd_${run_idx}"
        jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
