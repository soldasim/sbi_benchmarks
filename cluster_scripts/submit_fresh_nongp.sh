#!/bin/bash
# Fresh nongp runs: ProxySIRProblem 6-20, BananaProblem 20 — on cpu partition.
# Idempotent: skips if score_length==101 (done) or already queued.
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

    sbatch -p cpu --mem="$mem" --job-name="$job_name" \
        cluster_scripts/run.sh "$prob" nongp "$idx" 0 100 nothing
    ((submitted++))
}

echo "=== ProxySIRProblem nongp 6-20 ==="
for i in $(seq 6 20); do
    submit_job "ProxySIRProblem" "$i" "16G"
done

echo "=== BananaProblem nongp 20 ==="
submit_job "BananaProblem" "20" "8G"

echo "Done: submitted=$submitted  skipped_done=$skipped_done  skipped_queued=$skipped_queued"
