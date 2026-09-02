#!/bin/bash
# Extend DuffingProblem eiv/immd/nongp (currently 191-194/101/33-36 iters) and
# warpedgp-yja-maxvar (currently 90-96/100, post-samples-fix pilot) to a shared
# 200-iteration target, 24h wall time (cpu partition) per job.
#
# standard/est/loglike/loglike-imiqr already reached 201/201 - not resubmitted.
#
# All four go through script_continue_override.jl (data_dir workaround - see that
# file for why - warpedgp-yja needs it too, for an unrelated starts_dir/grid lookup
# bug in the plain script.jl path) plus cluster_scripts/jld2_compat_duffing_extend.jl
# (JLD2 migration shims for checkpoints saved under older BOSIP.jl/BOSS.jl struct
# layouts - see that file for the itemized list of what changed and why each shim
# is safe).
#
# Run from ~/repos/bosip_benchmarks.

# run_idx=1 for all four configs already submitted+running as debug/test jobs
# (11199105 eiv, 11199106 immd, 11199107 nongp, 11199108 warpedgp-yja) - this
# script covers the REMAINING indices only.

cd ~/repos/bosip_benchmarks

job_ids=()

echo "--- eiv (19 runs, data-bosip/DuffingProblem) ---"
for run_idx in $(seq 2 20); do
    job_name="DuffingProblem_eiv_${run_idx}_cont200"
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
        cluster_scripts/run_continue_override.sh DuffingProblem eiv "$run_idx" 200 nothing data-bosip/DuffingProblem)
    job_ids+=("$jid")
done

echo "--- immd (19 runs, data-bosip-norm/DuffingProblem) ---"
for run_idx in $(seq 2 20); do
    job_name="DuffingProblem_immd_${run_idx}_cont200"
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
        cluster_scripts/run_continue_override.sh DuffingProblem immd "$run_idx" 200 nothing data-bosip-norm/DuffingProblem)
    job_ids+=("$jid")
done

echo "--- nongp (19 runs, data-bosip/DuffingProblem) ---"
for run_idx in $(seq 2 20); do
    job_name="DuffingProblem_nongp_${run_idx}_cont200"
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
        cluster_scripts/run_continue_override.sh DuffingProblem nongp "$run_idx" 200 nothing data-bosip/DuffingProblem)
    job_ids+=("$jid")
done

echo "--- warpedgp-yja-maxvar (4 runs; own state in data-warpedgp2/ via warpedgp_data_dir, ---"
echo "    grid override points at data-bosip-norm/DuffingProblem where the grid lives) ---"
for run_idx in $(seq 2 5); do
    job_name="DuffingProblem_warpedgp-yja-maxvar_${run_idx}_cont200"
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
        cluster_scripts/run_continue_override.sh DuffingProblem warpedgp-yja-maxvar "$run_idx" 200 nothing data-bosip-norm/DuffingProblem)
    job_ids+=("$jid")
done

echo "Done. ${#job_ids[@]} jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
