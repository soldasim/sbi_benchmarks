#!/bin/bash
# Submit missing warpedgp-yja-maxvar jobs: SchafferN2_cross runs 3-20 and ThreeHumpCamel_cross runs 1-20.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

for run_idx in $(seq 3 20); do
    job_name="SchafferN2Problem_cross_wgp_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "SchafferN2Problem_cross" warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED"
done

for run_idx in $(seq 1 20); do
    job_name="ThreeHumpCamelProblem_cross_wgp_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "ThreeHumpCamelProblem_cross" warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED"
done

echo "Done. $count jobs submitted successfully."
echo "JOB_IDS: ${job_ids[*]}"
