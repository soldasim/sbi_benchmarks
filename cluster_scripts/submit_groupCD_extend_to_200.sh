#!/bin/bash
# Extend Group C `standard` runs (DuffingProblem5, DiffusionProblem5D) from
# 101 iters to the full 200-iter (201 incl. start) target, via main_continue.
# Uses cluster_scripts/run_continue_hd5d.sh / script_continue_hd5d.jl, NOT the
# plain run.sh/script.jl path - the standard checkpoints for these two
# problems predate BOSS.jl's MAPParams.loglike->logpost rename and JLD2 can't
# auto-reconstruct them without the migration shims (confirmed via 2 failed
# test jobs, 11314941/11314942, both MethodError on MAPParams{GaussianProcess}).
# Data: standard -> data-bosip-norm/<Problem>/
# Both problems have explicit data_dir overrides in data_paths.jl, so only the
# JLD2 shims are needed here, no data_dir override.
#
# The one short DiffusionProblem5D `warpedgp-yja-maxvar` run (run_idx=3,
# 199/201) was already submitted directly (job 11314943, via plain run.sh -
# its WarpedGaussianProcess checkpoint doesn't hit the MAPParams bug) and is
# NOT resubmitted by this script.
#
# run_idx=1 for both problems was already covered by test jobs
# 11314985 (DuffingProblem5) / 11314986 (DiffusionProblem5D) and is skipped here.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

for pname in DuffingProblem5 DiffusionProblem5D; do
    for run_idx in $(seq 2 20); do
        job_name="${pname}_standard_extend200_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
            cluster_scripts/run_continue_hd5d.sh "$pname" standard "$run_idx" 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
