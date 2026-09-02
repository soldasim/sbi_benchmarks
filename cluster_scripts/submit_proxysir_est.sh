#!/bin/bash
# Submit fresh est runs for ProxySIRProblem.
# 20 runs × 100 iters. Data stored in data-bosip-norm/ProxySIRProblem/.
# Idempotent: skips already-queued jobs and completed output files.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)
submitted=0; skipped_done=0; skipped_queued=0

for run_idx in $(seq 1 20); do
    job_name="ProxySIRProblem_est_${run_idx}"
    out_file="data-bosip-norm/ProxySIRProblem/est_${run_idx}_TVmetric.jld2"

    if [ -f "$out_file" ]; then ((skipped_done++)); continue; fi
    if echo "$queued" | grep -qx "$job_name"; then ((skipped_queued++)); continue; fi

    result=$(sbatch -p cpufast --mem=8G --job-name="$job_name" \
        cluster_scripts/run.sh ProxySIRProblem est "$run_idx" 0 100 nothing 2>&1)
    if echo "$result" | grep -q "error"; then
        echo "ERROR submitting $job_name: $result"
    else
        ((submitted++))
    fi
done

echo "Done: $skipped_done  Queued: $skipped_queued  New: $submitted"
