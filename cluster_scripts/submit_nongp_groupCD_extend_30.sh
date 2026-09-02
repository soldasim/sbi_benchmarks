#!/bin/bash
# Extend nongp (MaxVar acq + NonstatGP surrogate) runs for the Group C/D HD (5D)
# problems from their current (partition-wall-truncated) lengths up to 30 BO
# iterations (31 incl. start), via main_continue. DuffingProblem5 and
# MichalewiczProblem5_cross already exceed this target on every run and are
# skipped entirely. DiffusionProblem5D indices 6,8,10,11,13,16 are excluded:
# these crashed early in ConvergenceCallback (PosDefException, unfixed) and
# script.jl always passes convergence=true, so continuing them would just
# crash again immediately with no progress.
#
# cpu partition (1 day) — deltas needed are small (a handful of iterations
# for most runs), well within the 1-day limit given the observed ~13-14
# iters/day rate for these problems.
# Run from ~/repos/bosip_benchmarks.

cd ~/repos/bosip_benchmarks

TARGET_ITERS=30

declare -A run_idxs
# Note: idx 1 (Diffusion5D), idx 1 (Rosenbrock), idx 2 (StyblinskiTang), idx 16
# (Sphere) already submitted as debug-test jobs (11314979-11314982, confirmed
# healthy) and are excluded here to avoid a concurrent duplicate-write job.
run_idxs["DiffusionProblem5D"]="5 9 17 18"
run_idxs["RosenbrockProblem5_cross"]="2 3 4 5 6 7 9 10 11 13 14 15 16 17 18 19 20"
run_idxs["StyblinskiTangProblem5_cross"]="3 5 6 7 8 9 12 13 14 15 16 17 18"
run_idxs["SphereProblem5_cross"]="18"

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)

job_ids=()
count=0
skipped=0

submit_one() {
    local pname="$1" run_idx="$2"
    local job_name="${pname}_nongp_${run_idx}"
    if echo "$queued" | grep -qx "$job_name"; then
        echo "Skipping $job_name (already queued/running)"
        skipped=$((skipped + 1))
        return
    fi
    echo "Submitting $job_name (continue, target ${TARGET_ITERS} iters) ..."
    local jid
    jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" \
        cluster_scripts/run.sh "$pname" nongp "$run_idx" 1 "$TARGET_ITERS" nothing)
    job_ids+=("$jid")
    count=$((count + 1))
}

for pname in "${!run_idxs[@]}"; do
    for run_idx in ${run_idxs[$pname]}; do
        submit_one "$pname" "$run_idx"
    done
done

echo "Done. $count jobs submitted, $skipped skipped (already queued)."
echo "JOB_IDS: ${job_ids[*]}"
