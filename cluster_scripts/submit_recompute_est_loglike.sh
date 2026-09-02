#!/bin/bash
# Recompute TV metric scores for:
#   - est runs on the 7 standard problems (data-bosip/ → data-bosip-norm/)
#   - loglike and loglike-imiqr runs on the 7 Log problems (data-bosip/ → data-bosip-norm/)
#
# Uses log_approx_posterior (not log_posterior_mean) for these run types.
# Idempotent: skips completed output files and already-queued jobs.
#
# Prereq for Log problems: grids must exist in data-bosip-norm/LogXxxProblem/grid/

cd ~/repos/bosip_benchmarks

std_problems=("ABProblem" "SimpleProblem" "BananaProblem" "BimodalProblem" \
              "SIRProblem" "DuffingProblem" "DiffusionProblem10")
log_problems=("LogABProblem" "LogSimpleProblem" "LogBananaProblem" "LogBimodalProblem" \
              "LogSIRProblem" "LogDuffingProblem" "LogDiffusionProblem")

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)
submitted=0; skipped_done=0; skipped_queued=0; skipped_no_src=0; errors=0

submit_job() {
    local pname="$1" rname="$2" run_idx="$3"
    local job_name="${pname}_${rname}_tvnorm_${run_idx}"
    local src_file="data-bosip/${pname}/${rname}_${run_idx}_data.jld2"
    local out_file="data-bosip-norm/${pname}/${rname}_${run_idx}_TVmetric.jld2"

    [ ! -f "$src_file" ] && ((skipped_no_src++)) && return
    if [ -f "$out_file" ]; then ((skipped_done++)); return; fi
    if echo "$queued" | grep -qx "$job_name"; then ((skipped_queued++)); return; fi

    result=$(sbatch -p cpufast --mem=8G \
        --job-name="$job_name" \
        cluster_scripts/run_recompute.sh "$pname" "$rname" "$run_idx" 2>&1)
    if echo "$result" | grep -q "error"; then
        echo "ERROR submitting $job_name: $result"; ((errors++))
    else
        ((submitted++))
    fi
}

# est on 7 standard problems
for pname in "${std_problems[@]}"; do
    for run_idx in $(seq 1 20); do
        submit_job "$pname" "est" "$run_idx"
    done
done

# loglike and loglike-imiqr on 7 Log problems
for pname in "${log_problems[@]}"; do
    for rname in "loglike" "loglike-imiqr"; do
        for run_idx in $(seq 1 20); do
            submit_job "$pname" "$rname" "$run_idx"
        done
    done
done

echo "Done: $skipped_done  Queued: $skipped_queued  NoSrc: $skipped_no_src  New: $submitted  Errors: $errors"
