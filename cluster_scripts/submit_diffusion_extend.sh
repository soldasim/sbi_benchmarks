#!/bin/bash
# Extend DiffusionProblem10 standard/est/eiv/immd/warpedgp-yja-maxvar to a shared
# 200-iteration target, and nongp to 30 iters - plus LogDiffusionProblem
# loglike/loglike-imiqr to 200 iters. cpulong partition (3-day wall time) per job,
# per explicit user request.
#
# All go through script_continue_override_diffusion.jl (data_dir workaround for the
# original-7-BIP-problems routing split between data-bosip/ and data-bosip-norm/,
# same trap as DuffingProblem's extend-to-200 session - see that script's header)
# plus a post-load fix for Diffusion's simulator closure (captures the problem
# struct, unlike Duffing's stateless one - JLD2 loads it as a non-callable
# placeholder; script fixes it up by rebuilding BossProblem with a fresh
# simulator(problem) call) - see script_continue_override_diffusion.jl's header.
#
# idx=1 for all 8 configs already submitted+verified as test/debug jobs (11313791-8,
# retried as 11313804-5 after two bugs were found and fixed: warpedgp-yja-maxvar's
# checkpoint load path (uses warpedgp_data_dir, not data_dir) and a variable-scoping
# bug in main_loglike-imiqr.jl's convergence-callback try/catch, fixed at the source).
# This script covers the REMAINING indices only.
#
# LogDiffusionProblem loglike is special-cased: idx 3 and 17's problem checkpoints
# live in data-bosip-norm/ (not data-bosip/ like the other 18 indices) - confirmed
# via has_problem check before submission.
#
# Run from ~/repos/bosip_benchmarks.

cd ~/repos/bosip_benchmarks

job_ids=()

echo "--- DiffusionProblem10 standard (19 runs, data-bosip/, iters=200) ---"
for run_idx in $(seq 2 20); do
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="DiffusionProblem10_standard_${run_idx}_cont200" \
        cluster_scripts/run_continue_override_diffusion.sh DiffusionProblem10 standard "$run_idx" 200 nothing data-bosip/DiffusionProblem10)
    job_ids+=("$jid")
done

echo "--- DiffusionProblem10 est (19 runs, data-bosip/, iters=200) ---"
for run_idx in $(seq 2 20); do
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="DiffusionProblem10_est_${run_idx}_cont200" \
        cluster_scripts/run_continue_override_diffusion.sh DiffusionProblem10 est "$run_idx" 200 nothing data-bosip/DiffusionProblem10)
    job_ids+=("$jid")
done

echo "--- DiffusionProblem10 eiv (19 runs, data-bosip/, iters=200) ---"
for run_idx in $(seq 2 20); do
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="DiffusionProblem10_eiv_${run_idx}_cont200" \
        cluster_scripts/run_continue_override_diffusion.sh DiffusionProblem10 eiv "$run_idx" 200 nothing data-bosip/DiffusionProblem10)
    job_ids+=("$jid")
done

echo "--- DiffusionProblem10 nongp (19 runs, data-bosip/, iters=30) ---"
for run_idx in $(seq 2 20); do
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="DiffusionProblem10_nongp_${run_idx}_cont30" \
        cluster_scripts/run_continue_override_diffusion.sh DiffusionProblem10 nongp "$run_idx" 30 nothing data-bosip/DiffusionProblem10)
    job_ids+=("$jid")
done

echo "--- DiffusionProblem10 immd (19 runs, data-bosip-norm/, iters=200) ---"
for run_idx in $(seq 2 20); do
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="DiffusionProblem10_immd_${run_idx}_cont200" \
        cluster_scripts/run_continue_override_diffusion.sh DiffusionProblem10 immd "$run_idx" 200 nothing data-bosip-norm/DiffusionProblem10)
    job_ids+=("$jid")
done

echo "--- DiffusionProblem10 warpedgp-yja-maxvar (19 runs; own state via warpedgp_data_dir -> data-warpedgp2/, ---"
echo "    grid override points at data-bosip-norm/DiffusionProblem10 where the grid lives) ---"
for run_idx in $(seq 2 20); do
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="DiffusionProblem10_warpedgp-yja_${run_idx}_cont200" \
        cluster_scripts/run_continue_override_diffusion.sh DiffusionProblem10 warpedgp-yja-maxvar "$run_idx" 200 nothing data-bosip-norm/DiffusionProblem10)
    job_ids+=("$jid")
done

echo "--- LogDiffusionProblem loglike (19 runs, data-bosip/ except idx 3,17 -> data-bosip-norm/, iters=200) ---"
for run_idx in $(seq 2 20); do
    if [ "$run_idx" = "3" ] || [ "$run_idx" = "17" ]; then
        override_dir="data-bosip-norm/LogDiffusionProblem"
    else
        override_dir="data-bosip/LogDiffusionProblem"
    fi
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="LogDiffusionProblem_loglike_${run_idx}_cont200" \
        cluster_scripts/run_continue_override_diffusion.sh LogDiffusionProblem loglike "$run_idx" 200 nothing "$override_dir")
    job_ids+=("$jid")
done

echo "--- LogDiffusionProblem loglike-imiqr (19 runs, data-bosip/, iters=200) ---"
for run_idx in $(seq 2 20); do
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="LogDiffusionProblem_loglikeimiqr_${run_idx}_cont200" \
        cluster_scripts/run_continue_override_diffusion.sh LogDiffusionProblem loglike-imiqr "$run_idx" 200 nothing data-bosip/LogDiffusionProblem)
    job_ids+=("$jid")
done

echo "Done. ${#job_ids[@]} jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
