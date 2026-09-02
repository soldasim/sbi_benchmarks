#!/bin/bash
# Submit remaining IMMD runs:
#   - Runs 6–20 for the 7 original benchmark problems (ABProblem, SimpleProblem,
#     BananaProblem, BimodalProblem, ProxySIRProblem, DuffingProblem, DiffusionProblem10)
#   - Runs 1–20 for SIRProblem (new to IMMD experiments)
# Total: 15×7 + 20×1 = 125 jobs, 100 iters each.
# Data stored in data-bosip-norm/<ProblemName>/.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# Existing 7 problems — runs 6–20
for pname in ABProblem SimpleProblem BananaProblem BimodalProblem ProxySIRProblem; do
    for run_idx in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        job_name="${pname}_immd_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing
    done
done

for pname in DuffingProblem DiffusionProblem10; do
    for run_idx in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        job_name="${pname}_immd_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing
    done
done

# SIRProblem — runs 1–20 (all new)
for run_idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    job_name="SIRProblem_immd_${run_idx}"
    echo "Submitting $job_name ..."
    sbatch -p cpulong --mem=12G --job-name="$job_name" \
        cluster_scripts/run.sh SIRProblem immd "$run_idx" 0 100 nothing
done

echo "Done. 125 jobs submitted."
