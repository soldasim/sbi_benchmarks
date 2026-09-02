#!/bin/bash
# Fresh runs for missing individual indices on standard problems:
#   SimpleProblem eiv runs 10, 17
#   BananaProblem nongp run 20
# Idempotent: skips completed outputs and queued jobs.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)
submitted=0; skipped_done=0; skipped_queued=0; errors=0

submit_job() {
    local pname="$1" rname="$2" run_idx="$3" mem="$4"
    local job_name="${pname}_${rname}_${run_idx}"
    local out_file="data-bosip-norm/${pname}/${rname}_${run_idx}_TVmetric.jld2"

    if [ -f "$out_file" ]; then ((skipped_done++)); return; fi
    if echo "$queued" | grep -qx "$job_name"; then ((skipped_queued++)); return; fi

    result=$(sbatch -p cpufast --mem="$mem" --job-name="$job_name" \
        cluster_scripts/run.sh "$pname" "$rname" "$run_idx" 0 100 nothing 2>&1)
    if echo "$result" | grep -q "error"; then
        echo "ERROR submitting $job_name: $result"; ((errors++))
    else
        ((submitted++))
    fi
}

submit_job "SimpleProblem" "eiv"   10 "8G"
submit_job "SimpleProblem" "eiv"   17 "8G"
submit_job "BananaProblem" "nongp" 20 "16G"

echo "Done: $skipped_done  Queued: $skipped_queued  New: $submitted  Errors: $errors"
