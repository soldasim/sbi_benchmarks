#!/bin/bash
# Submit warpedgp-yja-maxvar runs for Groups C and D (after test job verified OK).
# Group C: DuffingProblem5 runs 1-20 + DiffusionProblem5D runs 1-20 = 40 jobs, cpulong, 200 iters.
#   (Run 1 for DuffingProblem5 overwrites the 100-iter test job data with proper 200-iter data.)
# Group D: 4 HD cross opt problems runs 1-20 = 80 jobs, cpulong, 200 iters.
# Total: 120 jobs. Data stored in data-warpedgp2/<ProblemName>/.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

# Group C — DuffingProblem5 runs 1-20 (run 1 overwrites 100-iter test with proper 200-iter data)
for run_idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    job_name="DuffingProblem5_warpedgp-yja_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
        cluster_scripts/run.sh DuffingProblem5 warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
done

# Group C — DiffusionProblem5D runs 1-20
for run_idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    job_name="DiffusionProblem5D_warpedgp-yja_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
        cluster_scripts/run.sh DiffusionProblem5D warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
done

# Group D — 4 HD cross opt problems, runs 1-20
group_d_problems=(
    "RosenbrockProblem5_cross"
    "StyblinskiTangProblem5_cross"
    "MichalewiczProblem5_cross"
    "SphereProblem5_cross"
)
for pname in "${group_d_problems[@]}"; do
    for run_idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        job_name="${pname}_warpedgp-yja_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
