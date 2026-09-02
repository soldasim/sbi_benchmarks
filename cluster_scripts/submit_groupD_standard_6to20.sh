#!/bin/bash
# Submit standard (maxvar) runs 6-20 for Group D (4 HD cross opt problems).
# 4 problems × 15 runs = 60 jobs, cpu partition.
# Data stored in data-opt-functions/<ProblemName>/.

cd ~/repos/bosip_benchmarks

problems=(
    "RosenbrockProblem5_cross"
    "StyblinskiTangProblem5_cross"
    "MichalewiczProblem5_cross"
    "SphereProblem5_cross"
)

job_ids=()
count=0
for pname in "${problems[@]}"; do
    for run_idx in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        job_name="${pname}_maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" maxvar "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
