#!/bin/bash
# Recompute TV-metric scores for every Group B (problem, run_name, run_idx) combo currently
# NaN-polluted, using the "stabilized" posterior estimator (log_posterior_mean_safe,
# src/recompute_tv_general_safe.jl / src/safe_posterior_estimator.jl). Refits the run's own
# model type (GaussianProcess for maxvar/eiv, NonstationaryGP for nongp) on the existing
# _data.jld2 observation trajectory — no new simulator evaluations, no BO/acquisition loop.
#
# Writes to a NEW file (<run_name>_<idx>_TVmetric_safe.jld2) — never overwrites the original
# _TVmetric.jld2, so old (NaN-containing) scores remain available for comparison.
#
# Target list: cluster_scripts/groupb_nan_target_runs.csv (problem,run_name,run_idx,nan_count,len)
# — every row is a run with nan_count > 0, from a fresh audit (2026-08-14) of all 24 Group B
# cross-polytope 2D problems x {maxvar, eiv, nongp} (warpedgp-yja-maxvar and immd were confirmed
# fully clean and are NOT in this list).
#
# Partition: `cpu` (1 day) for all — recompute skips the acquisition-optimization step that
# dominates the original runs' cost (especially eiv's integrated-variance acquisition), so it
# is expected to be substantially cheaper than the original run's wall-clock time.
#
# Idempotent: skips a row if its job is already queued, or if a _TVmetric_safe.jld2 with the
# same length as the original (NaN-containing) score already exists.
#
# Run from ~/repos/bosip_benchmarks (on the cluster, via ssh — NOT the login node itself for
# the actual Julia work, sbatch only submits, doesn't run Julia on the login node).

cd ~/repos/bosip_benchmarks

csv="cluster_scripts/groupb_nan_target_runs.csv"
if [ ! -f "$csv" ]; then
    echo "Target list not found: $csv"
    exit 1
fi

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)

submitted=0
skipped_done=0
skipped_queued=0

while IFS=, read -r prob run_name run_idx nan_count len; do
    [ -z "$prob" ] && continue

    job_name="${prob}_${run_name}_${run_idx}_tvsafe"

    if echo "$queued" | grep -qx "$job_name"; then
        ((skipped_queued++))
        continue
    fi

    data_dir="data-opt-functions/${prob}"
    safe_file="${data_dir}/${run_name}_${run_idx}_TVmetric_safe.jld2"
    if [ -f "$safe_file" ]; then
        safe_len=$(~/.juliaup/bin/julialauncher --project=src -e "
            using JLD2
            print(length(load(\"$safe_file\", \"score\")))
        " 2>/dev/null)
        if [ "$safe_len" = "$len" ] && [ -n "$safe_len" ]; then
            ((skipped_done++))
            continue
        fi
    fi

    sbatch -p cpu --mem=12G --job-name="$job_name" \
        cluster_scripts/run_recompute_general_safe.sh "$prob" "$run_name" "$run_idx"
    ((submitted++))
done < "$csv"

echo "Done: submitted=$submitted  skipped_done=$skipped_done  skipped_queued=$skipped_queued"
