#!/bin/bash
# Continue batch 4 and 5 runs on appropriate partitions.
# Batch 4: ProxySIRProblem eiv/nongp runs 6-20 (continue=1)
#   eiv  → cpulong (62h/run), 8G
#   nongp → cpu    (15h/run), 16G
# Batch 5: SimpleProblem eiv_10/17, BananaProblem nongp_20 (continue=1)
#   eiv  → cpu (5h remaining), 8G
#   nongp → cpu (8h remaining), 8G
#
# Idempotent: skips runs with score_length==101 (done) or already queued.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)
submitted=0; skipped_done=0; skipped_queued=0

submit_job() {
    local prob="$1" rname="$2" idx="$3" part="$4" mem="$5"
    local job_name="${prob}_${rname}_${idx}"

    # Check if job is already queued
    if echo "$queued" | grep -qx "$job_name"; then
        ((skipped_queued++)); return
    fi

    # Check if score is complete (101 entries = 100 iters done)
    local score_file="data-bosip-norm/${prob}/${rname}_${idx}_TVmetric.jld2"
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

    sbatch -p "$part" --mem="$mem" --job-name="$job_name" \
        cluster_scripts/run.sh "$prob" "$rname" "$idx" 1 100 nothing
    ((submitted++))
}

echo "=== ProxySIRProblem eiv 6-20 (cpulong) ==="
for i in $(seq 6 20); do
    submit_job "ProxySIRProblem" "eiv" "$i" "cpulong" "8G"
done

echo "=== ProxySIRProblem nongp 6-20 (cpu) ==="
for i in $(seq 6 20); do
    submit_job "ProxySIRProblem" "nongp" "$i" "cpu" "16G"
done

echo "=== SimpleProblem eiv 10, 17 (cpu) ==="
submit_job "SimpleProblem" "eiv" "10" "cpu" "8G"
submit_job "SimpleProblem" "eiv" "17" "cpu" "8G"

echo "=== BananaProblem nongp 20 (cpu) ==="
submit_job "BananaProblem" "nongp" "20" "cpu" "8G"

echo "Done: submitted=$submitted  skipped_done=$skipped_done  skipped_queued=$skipped_queued"
