#!/bin/bash
# Fix missing/partial maxvar runs for BealeProblem_cross and GoldsteinPriceProblem_cross.
# Audit (2026-07-01, job 11110004) showed:
#   - BealeProblem_cross: runs 1–5 missing, runs 13(82)/16(38)/17(9) < 100 iters
#   - GoldsteinPriceProblem_cross: runs 1–5 missing; runs 6–20 already >= 100 iters
# Target: 100 iters (2D rule). Partition: cpu.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

submit() {
    local pname="$1" run_idx="$2"
    local job_name="${pname}_cross_maxvar_${run_idx}"
    jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" \
          cluster_scripts/run.sh "${pname}_cross" maxvar "$run_idx" 0 100 nothing)
    if [ -n "$jid" ]; then
        job_ids+=("$jid")
        count=$((count + 1))
    else
        echo "FAILED: $job_name"
    fi
}

# BealeProblem_cross: runs 1–5 missing + runs 13,16,17 partial (< 100 iters)
for run_idx in 1 2 3 4 5 13 16 17; do
    submit "BealeProblem" "$run_idx"
done

# GoldsteinPriceProblem_cross: runs 1–5 missing
for run_idx in 1 2 3 4 5; do
    submit "GoldsteinPriceProblem" "$run_idx"
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
