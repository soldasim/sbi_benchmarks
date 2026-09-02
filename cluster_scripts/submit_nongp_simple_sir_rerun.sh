#!/bin/bash
# Rerun crashed nongp (NonstationaryGP) runs on SimpleProblem and SIRProblem.
# These crashed with PosDefException in NonstationaryGP's own covariance
# construction (finite_nongp) — BOSS.jl commit 5e57410 (2026-07-16) added a
# jitter-retry mechanism to that exact path, which this rerun tests at scale.
#
# SimpleProblem: all 20 runs crashed originally -> rerun 1-20.
# SIRProblem: runs 1-2 timed out (valid, not crashed) -> rerun 3-20 only.
# Fresh runs (continue=0), cpulong partition (3 days), 100 iters target.
#
# Idempotent: skips runs with score_length==101 (done) or already queued.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)
submitted=0; skipped_done=0; skipped_queued=0

submit_job() {
    local prob="$1" idx="$2" mem="$3"
    local job_name="${prob}_nongp_${idx}"

    if echo "$queued" | grep -qx "$job_name"; then
        ((skipped_queued++)); return
    fi

    local score_file="data-bosip-norm/${prob}/nongp_${idx}_TVmetric.jld2"
    if [ -f "$score_file" ]; then
        local slen
        slen=$(~/.juliaup/bin/julialauncher --project=src -e "
            using JLD2
            s = load(\"$score_file\", \"score\")
            print(isnothing(s) ? 0 : length(s))
        " 2>/dev/null)
        if [ "$slen" = "101" ]; then
            ((skipped_done++)); return
        fi
    fi

    sbatch -p cpulong --mem="$mem" --job-name="$job_name" \
        cluster_scripts/run.sh "$prob" nongp "$idx" 0 100 nothing
    ((submitted++))
}

echo "=== SimpleProblem nongp 1-20 ==="
for i in $(seq 1 20); do
    submit_job "SimpleProblem" "$i" "8G"
done

echo "=== SIRProblem nongp 3-20 ==="
for i in $(seq 3 20); do
    submit_job "SIRProblem" "$i" "8G"
done

echo "Done: submitted=$submitted  skipped_done=$skipped_done  skipped_queued=$skipped_queued"
