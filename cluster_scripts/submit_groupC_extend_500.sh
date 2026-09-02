#!/bin/bash
# Extend runs 1-5 (only) of `standard`/`maxvar` from 200 to 500 iters, for all
# 6 merged-Group-C HD problems. Per explicit user request (2026-08-07):
# supplementary/exploratory data on a subset of runs, not part of the
# currently-scoped Plot 2c/3c figures (which only require the 200-iter target,
# already met for all 20 runs, see project_bosip_paper_plan.md).
#
# ALL 6 problems route through cluster_scripts/run_continue_hd5d.sh, not plain
# run.sh - every one of these checkpoints predates BOSS.jl's
# MAPParams.loglike->logpost rename (confirmed via failed test job 11315889 on
# RosenbrockProblem5_cross/maxvar, same MethodError as the earlier
# standard/DuffingProblem5+DiffusionProblem5D 200-iter extension), regardless
# of run_name. run_continue_hd5d.sh is generic over run_name (includes
# main_<run_name>.jl dynamically) so this works unchanged for maxvar too.
#
# run_idx=1 for all 6 problems already covered by debug test jobs:
#   DuffingProblem5 standard: 11315888
#   RosenbrockProblem5_cross maxvar: 11315891 (11315889 failed on plain run.sh, retried)
#   StyblinskiTangProblem5_cross/MichalewiczProblem5_cross/SphereProblem5_cross/
#   DiffusionProblem5D run 1: NOT yet submitted by a dedicated test - included
#   in the bulk loop below since run_continue_hd5d.sh is now proven generic
#   across both run_name and problem type.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

# BIP problems (data-bosip-norm/), run_name=standard, mem=12G
for pname in DuffingProblem5 DiffusionProblem5D; do
    for run_idx in $(seq 1 5); do
        # skip DuffingProblem5 run 1 - already running as test job 11315888
        if [[ "$pname" == "DuffingProblem5" && "$run_idx" == "1" ]]; then
            echo "Skipping DuffingProblem5 run 1 (already submitted as test job 11315888)"
            continue
        fi
        job_name="${pname}_standard_extend500_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run_continue_hd5d.sh "$pname" standard "$run_idx" 500 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

# Cross opt-function problems (data-opt-functions/<name>_cross/), run_name=maxvar, mem=16G
for pname in RosenbrockProblem5_cross StyblinskiTangProblem5_cross MichalewiczProblem5_cross SphereProblem5_cross; do
    for run_idx in $(seq 1 5); do
        # skip RosenbrockProblem5_cross run 1 - already running as test job 11315891
        if [[ "$pname" == "RosenbrockProblem5_cross" && "$run_idx" == "1" ]]; then
            echo "Skipping RosenbrockProblem5_cross run 1 (already submitted as test job 11315891)"
            continue
        fi
        job_name="${pname}_maxvar_extend500_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
            cluster_scripts/run_continue_hd5d.sh "$pname" maxvar "$run_idx" 500 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
