#!/bin/bash
# Fresh eiv and nongp runs for ProxySIRProblem, runs 6-20.
# Runs 1-5 already exist. Idempotent: skips completed outputs and queued jobs.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)
submitted=0; skipped_done=0; skipped_queued=0; errors=0

submit_job() {
    local rname="$1" run_idx="$2" mem="$3"
    local job_name="ProxySIRProblem_${rname}_${run_idx}"
    local out_file="data-bosip-norm/ProxySIRProblem/${rname}_${run_idx}_TVmetric.jld2"

    if [ -f "$out_file" ]; then ((skipped_done++)); return; fi
    if echo "$queued" | grep -qx "$job_name"; then ((skipped_queued++)); return; fi

    result=$(sbatch -p cpufast --mem="$mem" --job-name="$job_name" \
        cluster_scripts/run.sh ProxySIRProblem "$rname" "$run_idx" 0 100 nothing 2>&1)
    if echo "$result" | grep -q "error"; then
        echo "ERROR submitting $job_name: $result"; ((errors++))
    else
        ((submitted++))
    fi
}

for run_idx in $(seq 6 20); do
    submit_job "eiv"   "$run_idx" "8G"
    submit_job "nongp" "$run_idx" "16G"
done

echo "Done: $skipped_done  Queued: $skipped_queued  New: $submitted  Errors: $errors"
