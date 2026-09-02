#!/bin/bash
# Submit nongp (MaxVar acq + NonstatGP surrogate) runs 1-20 for the 6 HD (5D)
# problems: Group C (2 HD BIP) + Group D (4 HD cross opt). None of these
# problems have any nongp data yet (verified 2026-07-24).
#
# cpulong partition (3 days, HD nongp is expensive), 200 iters.
# Submit AFTER Group B's batch (submit_nongp_groupB_1to20.sh) per explicit
# user ordering instruction. Run from ~/repos/bosip_benchmarks.

cd ~/repos/bosip_benchmarks

# Group C — HD BIP problems, data in data-bosip-norm/
group_c=(
    "DuffingProblem5"
    "DiffusionProblem5D"
)

# Group D — HD cross-polytope opt problems, data in data-opt-functions/
group_d=(
    "RosenbrockProblem5_cross"
    "StyblinskiTangProblem5_cross"
    "MichalewiczProblem5_cross"
    "SphereProblem5_cross"
)

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)

job_ids=()
count=0
skipped=0

submit_one() {
    local pname="$1" run_idx="$2"
    local job_name="${pname}_nongp_${run_idx}"
    if echo "$queued" | grep -qx "$job_name"; then
        echo "Skipping $job_name (already queued/running, e.g. a debug-test job)"
        skipped=$((skipped + 1))
        return
    fi
    echo "Submitting $job_name ..."
    local jid
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
        cluster_scripts/run.sh "$pname" nongp "$run_idx" 0 200 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
}

echo "=== Group C (HD BIP) ==="
for pname in "${group_c[@]}"; do
    for run_idx in $(seq 1 20); do
        submit_one "$pname" "$run_idx"
    done
done

echo "=== Group D (HD cross opt) ==="
for pname in "${group_d[@]}"; do
    for run_idx in $(seq 1 20); do
        submit_one "$pname" "$run_idx"
    done
done

echo "Done. $count jobs submitted, $skipped skipped (already queued)."
echo "JOB_IDS: ${job_ids[*]}"
