#!/bin/bash
# Fresh BO runs for Log problem indices missing from source data-bosip/.
# These indices never existed; runs write directly to data-bosip-norm/.
# Start files have been copied to data-bosip-norm/<problem>/starts/.
# Idempotent: skips completed outputs and already-queued jobs.
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
        echo "ERROR: $job_name: $result"; ((errors++))
    else
        ((submitted++))
    fi
}

submit_job "LogSimpleProblem"    "loglike-imiqr" 9  "8G"
submit_job "LogBananaProblem"    "loglike-imiqr" 10 "8G"
submit_job "LogDiffusionProblem" "loglike"        3  "8G"
submit_job "LogDiffusionProblem" "loglike"        17 "8G"

echo "Done: $skipped_done  Queued: $skipped_queued  New: $submitted  Errors: $errors"
