#!/bin/bash
# Submit final 9 missing warpedgp-yja-maxvar jobs:
#   ThreeHumpCamelProblem_cross runs 12-20 (runs 1-11 already submitted)

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

for run_idx in $(seq 12 20); do
    job_name="ThreeHumpCamelProblem_cross_wgp_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "ThreeHumpCamelProblem_cross" warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"
done

echo "Done. $count jobs submitted successfully."
echo "JOB_IDS: ${job_ids[*]}"
