#!/bin/bash
# Continue eiv/immd runs on the 6 merged-Group-C 5D problems toward the
# revised (2026-08-06) lowered targets: eiv -> 30 iters, immd -> 100 iters.
# Only 5 runs per config (runs 1-5), per the project's HD-problem rule.
#
# Fresh on-disk audit (this session) showed some (problem, config) pairs
# already meet the new target - those are NOT included below:
#   - DuffingProblem5 eiv: already 30-32, skipped entirely.
#   - RosenbrockProblem5_cross immd: already 104-107, skipped entirely.
#   - MichalewiczProblem5_cross immd: already 101-109, skipped entirely.
#   - StyblinskiTangProblem5_cross immd run 3: already 102, skipped (only
#     runs 1,2,4,5 continued for this one).
#
# No data_dir *routing* bug applies to these 6 problems (unlike the legacy 7
# BIP problems) - DuffingProblem5/DiffusionProblem5D already route correctly
# to data-bosip-norm/, the 4 cross opt-function problems to data-opt-functions/.
#
# HOWEVER: debug test jobs (11315200-11315203, run via plain run.sh + continue=1)
# all crashed identically in main_continue's `load(...)` of the old *_problem.jld2
# checkpoint: `MAPParams{GaussianProcess}` is missing the `logpost` field, the
# exact same BOSS.jl `5f06d33` rename issue already solved for DuffingProblem's
# checkpoints (see jld2_compat_duffing_extend.jl). So we route through
# script_continue_override.jl instead (which already includes that shim file),
# passing each problem's ALREADY-correct data_dir as a true no-op override -
# reuses proven infrastructure instead of writing new migration code.
#
# Run from ~/repos/bosip_benchmarks.

cd ~/repos/bosip_benchmarks

job_ids=()

submit() {
    local pname=$1 run_name=$2 run_idx=$3 iters=$4 mem=$5 suffix=$6 override_dir=$7
    local job_name="${pname}_${run_name}_${run_idx}_${suffix}"
    jid=$(sbatch --parsable -p cpulong --mem="$mem" --job-name="$job_name" \
        cluster_scripts/run_continue_override.sh "$pname" "$run_name" "$run_idx" "$iters" nothing "$override_dir")
    job_ids+=("$jid")
    echo "Submitted $job_name -> job $jid"
}

override_dir_for() {
    local pname=$1
    case "$pname" in
        DuffingProblem5|DiffusionProblem5D) echo "data-bosip-norm/${pname}" ;;
        *_cross) echo "data-opt-functions/${pname}" ;;
        *) echo "ERROR-UNKNOWN-PROBLEM-$pname" ;;
    esac
}

# ---- DEBUG TEST JOBS (one per BIP/opt-function x eiv/immd combo) ----
if [[ "$1" == "test" ]]; then
    echo "--- Submitting 4 debug test jobs only (via script_continue_override.jl, no-op data_dir override) ---"
    submit DiffusionProblem5D eiv 1 30 12G cont30_TEST2 "$(override_dir_for DiffusionProblem5D)"
    submit DuffingProblem5 immd 1 100 12G cont100_TEST2 "$(override_dir_for DuffingProblem5)"
    submit RosenbrockProblem5_cross eiv 1 30 16G cont30_TEST2 "$(override_dir_for RosenbrockProblem5_cross)"
    submit StyblinskiTangProblem5_cross immd 1 100 16G cont100_TEST2 "$(override_dir_for StyblinskiTangProblem5_cross)"
    echo "Done. ${#job_ids[@]} test jobs submitted."
    echo "JOB_IDS: ${job_ids[*]}"
    exit 0
fi

# ---- BULK (remaining runs, excludes the 4 already-submitted test-job indices) ----
echo "--- eiv, target 30 iters ---"
for pname in DiffusionProblem5D RosenbrockProblem5_cross StyblinskiTangProblem5_cross MichalewiczProblem5_cross SphereProblem5_cross; do
    mem=12G
    [[ "$pname" == *_cross ]] && mem=16G
    odir="$(override_dir_for "$pname")"
    for run_idx in 1 2 3 4 5; do
        # skip DiffusionProblem5D run1 and RosenbrockProblem5_cross run1 (already covered by test jobs)
        if [[ "$pname" == "DiffusionProblem5D" && "$run_idx" == "1" ]]; then continue; fi
        if [[ "$pname" == "RosenbrockProblem5_cross" && "$run_idx" == "1" ]]; then continue; fi
        submit "$pname" eiv "$run_idx" 30 "$mem" cont30 "$odir"
    done
done

echo "--- immd, target 100 iters ---"
odir="$(override_dir_for DuffingProblem5)"
for run_idx in 1 2 3 4 5; do
    if [[ "$run_idx" == "1" ]]; then continue; fi  # DuffingProblem5 immd run1 covered by test job
    submit DuffingProblem5 immd "$run_idx" 100 12G cont100 "$odir"
done
odir="$(override_dir_for DiffusionProblem5D)"
for run_idx in 1 2 3 4 5; do
    submit DiffusionProblem5D immd "$run_idx" 100 12G cont100 "$odir"
done
odir="$(override_dir_for StyblinskiTangProblem5_cross)"
for run_idx in 1 2 4 5; do
    if [[ "$run_idx" == "1" ]]; then continue; fi  # StyblinskiTangProblem5_cross immd run1 covered by test job
    submit StyblinskiTangProblem5_cross immd "$run_idx" 100 16G cont100 "$odir"
done
odir="$(override_dir_for SphereProblem5_cross)"
for run_idx in 1 2 3 4 5; do
    submit SphereProblem5_cross immd "$run_idx" 100 16G cont100 "$odir"
done

echo "Done. ${#job_ids[@]} jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
