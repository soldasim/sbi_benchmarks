#!/bin/bash
# Continue DuffingProblem immd (currently 178-197/200) to the full 200-iteration
# target, 24h wall time (cpu partition). EIV was checked and found already
# complete at 201/201 in data-bosip/DuffingProblem (the 191-194 copy in
# data-bosip-norm was a stale non-continuable snapshot, no _problem.jld2) - not
# resubmitted. Reuses the run_continue_override.sh / script_continue_override.jl
# pipeline from the 2026-07-23 DuffingProblem extension.
#
# Run from ~/repos/bosip_benchmarks.

cd ~/repos/bosip_benchmarks

job_ids=()

echo "--- immd (20 runs, data-bosip-norm/DuffingProblem) ---"
for run_idx in $(seq 1 20); do
    job_name="DuffingProblem_immd_${run_idx}_cont200v2"
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
        cluster_scripts/run_continue_override.sh DuffingProblem immd "$run_idx" 200 nothing data-bosip-norm/DuffingProblem)
    job_ids+=("$jid")
done

echo "Done. ${#job_ids[@]} jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
