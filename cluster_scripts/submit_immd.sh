#!/bin/bash
# Submit the 7 original benchmark problems with the updated IMMD acquisition.
# 5 runs × 7 problems = 35 jobs, 100 iters each.
# Data stored in data-bosip-norm/<ProblemName>/.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# Analytical problems (2D) — 3 days should be sufficient
for pname in ABProblem SimpleProblem BananaProblem BimodalProblem ProxySIRProblem; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_immd_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --time=3-00:00:00 --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing
    done
done

# PDE problems — cpulong max is 3 days
for pname in DuffingProblem DiffusionProblem10; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_immd_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --time=3-00:00:00 --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing
    done
done

echo "Done. 35 jobs submitted."
