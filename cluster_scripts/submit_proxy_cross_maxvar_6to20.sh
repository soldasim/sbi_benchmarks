#!/bin/bash
# Submit maxvar runs 6-20 for BealeProxyProblem_cross and GoldsteinPriceProxyProblem_cross
# (with-proxy Group B problems). Runs 1-5 already exist and are clean (201 iters, 0 NaN).
# Mirrors submit_cross2d_maxvar_6to20.sh (non-proxy siblings): same partition/mem/iters.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

problems=(
    "BealeProxyProblem_cross"
    "GoldsteinPriceProxyProblem_cross"
)

job_ids=()
count=0
for pname in "${problems[@]}"; do
    for run_idx in $(seq 6 20); do
        job_name="${pname}_maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" maxvar "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
