#!/bin/bash
# Submit warpedgp-yja-maxvar runs 1-20 for ProxySIRProblem (Group A, with-proxy).
# Mirrors the Group A samplesfix pilot (submit_warpedgp_yja_groupA_samplesfix_pilot.sh):
# same partition/mem/iters, just a different (with-proxy) problem.
# No prior warpedgp-yja-maxvar data exists for this problem (0/20 before this script).
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

job_ids=()
count=0
for run_idx in $(seq 1 20); do
    job_name="ProxySIRProblem_warpedgp-yja-maxvar_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" cluster_scripts/run.sh "ProxySIRProblem" warpedgp-yja-maxvar "$run_idx" 0 100 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
