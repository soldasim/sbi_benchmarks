#!/bin/bash
# Recompute nongp TV-metric scores for SimpleProblem/SIRProblem using log_posterior_mean_safe,
# by re-fitting NonstationaryGP on each recorded data-iteration prefix (src/recompute_tv_safe.jl).
# Writes to a NEW file (`_TVmetric_safe.jld2`) — never overwrites the original `_TVmetric.jld2`.
#
# SimpleProblem: all 20 runs.
# SIRProblem: runs 3-20 only — runs 1-2 are currently being rerun by a separate job
# (SIRProblem_nongp_1/2, expected to finish ~2026-08-09) and must not be touched concurrently.
# Submit those two once that rerun completes.
#
# cpulong partition (3 days), 8G mem — matches the original nongp run resource profile.
# Idempotent: skips if already queued or if the _safe output already has the target length.
#
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)
submitted=0; skipped_done=0; skipped_queued=0

submit_job() {
    local prob="$1" idx="$2"
    local job_name="${prob}_nongp_${idx}_tvsafe"

    if echo "$queued" | grep -qx "$job_name"; then
        ((skipped_queued++)); return
    fi

    local target_file="data-bosip-norm/${prob}/nongp_${idx}_TVmetric.jld2"
    local safe_file="data-bosip-norm/${prob}/nongp_${idx}_TVmetric_safe.jld2"
    if [ -f "$target_file" ] && [ -f "$safe_file" ]; then
        local target_len safe_len
        target_len=$(~/.juliaup/bin/julialauncher --project=src -e "
            using JLD2
            print(length(load(\"$target_file\", \"score\")))
        " 2>/dev/null)
        safe_len=$(~/.juliaup/bin/julialauncher --project=src -e "
            using JLD2
            print(length(load(\"$safe_file\", \"score\")))
        " 2>/dev/null)
        if [ "$safe_len" = "$target_len" ] && [ -n "$safe_len" ]; then
            ((skipped_done++)); return
        fi
    fi

    sbatch -p cpulong --mem=8G --job-name="$job_name" \
        cluster_scripts/run_recompute_tv_safe.sh "$prob" nongp "$idx"
    ((submitted++))
}

echo "=== SimpleProblem nongp 1-20 (TV recompute) ==="
for i in $(seq 1 20); do
    submit_job "SimpleProblem" "$i"
done

echo "=== SIRProblem nongp 3-20 (TV recompute; 1-2 excluded, still being rerun) ==="
for i in $(seq 3 20); do
    submit_job "SIRProblem" "$i"
done

echo "Done: submitted=$submitted  skipped_done=$skipped_done  skipped_queued=$skipped_queued"
