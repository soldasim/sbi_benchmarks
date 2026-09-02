#!/bin/bash
# Submit immd runs 1-20 for BealeProxyProblem_cross and GoldsteinPriceProxyProblem_cross
# (with-proxy Group B problems). No prior immd data exists for these problems (0/20 before
# this script). Mirrors submit_cross2d_immd_1to20.sh (non-proxy siblings): same
# partition/mem/iters (100 iters is the established Group B immd target, not 200).
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

problems=(
    "BealeProxyProblem_cross"
    "GoldsteinPriceProxyProblem_cross"
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
