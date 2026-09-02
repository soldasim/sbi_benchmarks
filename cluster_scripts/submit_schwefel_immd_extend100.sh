#!/bin/bash
# Continue SchwefelProblem2_cross immd runs that are short of the 100-iter
# (101-entry) target on Group B, per project_bosip_paper_plan.md's 2026-08-06
# revised Rules (Group B eiv/immd target = 100 iters, down from 200).
# These runs previously timed out at 24h on `cpu`; per user instruction, give
# all runs a 3-day (`cpulong`) budget this time. continue=1 (existing
# checkpoints), iters=100 (target total, not incremental).

cd ~/repos/bosip_benchmarks

run_indices=(1 3 5 6 7 9 10 12 14 15 16 19 20)

job_ids=()
count=0
for run_idx in "${run_indices[@]}"; do
    job_name="SchwefelProblem2_cross_immd_${run_idx}_extend100"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
        cluster_scripts/run.sh "SchwefelProblem2_cross" immd "$run_idx" 1 100 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
