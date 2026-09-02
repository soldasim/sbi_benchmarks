#!/bin/bash
# Submit IMMD runs 1-20 for Group D (4 HD cross opt problems).
# 4 problems × 20 runs = 80 jobs, cpulong partition.
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
    for run_idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        job_name="${pname}_immd_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" immd "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
