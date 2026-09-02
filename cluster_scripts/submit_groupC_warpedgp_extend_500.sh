#!/bin/bash
# Extend runs 1-5 (only) of `warpedgp-yja-maxvar` from 200 to 500 iters, for
# all 6 merged-Group-C HD problems. Per explicit user request (2026-08-07),
# mirrors submit_groupC_extend_500.sh's standard/maxvar 500-iter extension.
# Deliberately EXCLUDES the newly-added Ackley5D problem - being handled by a
# different session, do not touch it here.
#
# All 6 problems route through cluster_scripts/run_continue_hd5d.sh - proven
# generic over run_name (dynamically includes main_<run_name>.jl), so the
# same infrastructure written for standard/maxvar works unchanged here.
# warpedgp-yja-maxvar checkpoints (MAPParams{WarpedGaussianProcess}) are not
# expected to hit the MAPParams JLD2 rename bug that hit standard/maxvar
# (MAPParams{GaussianProcess}) - confirmed via 2 clean debug test jobs
# (11317609 DuffingProblem5, 11317610 RosenbrockProblem5_cross) before this
# bulk submission.
#
# run_idx=1 for DuffingProblem5 and RosenbrockProblem5_cross already covered
# by the debug test jobs above and skipped here.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

# BIP problems (data-warpedgp2/<name>/ via warpedgp_data_dir), mem=12G
for pname in DuffingProblem5 DiffusionProblem5D; do
    for run_idx in $(seq 1 5); do
        if [[ "$pname" == "DuffingProblem5" && "$run_idx" == "1" ]]; then
            echo "Skipping DuffingProblem5 run 1 (already submitted as test job 11317609)"
            continue
        fi
        job_name="${pname}_warpedgp-yja-maxvar_extend500_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run_continue_hd5d.sh "$pname" warpedgp-yja-maxvar "$run_idx" 500 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

# Cross opt-function problems (data-warpedgp2/<name>_cross/), mem=16G
for pname in RosenbrockProblem5_cross StyblinskiTangProblem5_cross MichalewiczProblem5_cross SphereProblem5_cross; do
    for run_idx in $(seq 1 5); do
        if [[ "$pname" == "RosenbrockProblem5_cross" && "$run_idx" == "1" ]]; then
            echo "Skipping RosenbrockProblem5_cross run 1 (already submitted as test job 11317610)"
            continue
        fi
        job_name="${pname}_warpedgp-yja-maxvar_extend500_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
            cluster_scripts/run_continue_hd5d.sh "$pname" warpedgp-yja-maxvar "$run_idx" 500 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
