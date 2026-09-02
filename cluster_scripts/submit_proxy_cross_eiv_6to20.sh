#!/bin/bash
# Submit eiv runs 6-20 for BealeProxyProblem_cross and GoldsteinPriceProxyProblem_cross
# (with-proxy Group B problems). Runs 1-5 already exist (176-188 iters, timed out; one run
# of GoldsteinPriceProxyProblem_cross has 33 NaN, accepted per iteration policy).
# Mirrors submit_cross2d_eiv_6to20.sh (non-proxy siblings): same partition/mem/iters.
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
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
