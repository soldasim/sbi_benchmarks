#!/bin/bash
# Submit fresh EIV runs 6-25 for Group D (4 HD cross opt problems).
# Indices 6-25 avoid collision with unusable existing files at indices 1-5.
# 4 problems × 20 runs = 80 jobs, cpulong partition (~45h per run at 5D).
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
    for run_idx in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
