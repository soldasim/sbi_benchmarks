#!/bin/bash
# Submit standard runs 6-20 for Group C (2 HD BIP problems).
# 2 problems × 15 runs = 30 jobs, cpulong partition.
# Data stored in data-bosip-norm/<ProblemName>/.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0
for pname in DuffingProblem5 DiffusionProblem5D; do
    for run_idx in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        job_name="${pname}_standard_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" standard "$run_idx" 0 100 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
