#!/bin/bash
# Submit WarpedGaussianProcess + LogMaxVar on the 7 original benchmark problems.
# WarpedGP uses adaptive YeoJohnson output warping as a learned proxy variable.
# Compare against submit_immd.sh (plain GP variants) for the same problems.
# 5 runs × 7 problems = 35 jobs, 100 iters each.
# Data stored in data-bosip-norm/<ProblemName>/.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# Analytical problems (2D)
for pname in ABProblem SimpleProblem BananaProblem BimodalProblem ProxySIRProblem; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_warpedgp-maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --time=3-00:00:00 --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-maxvar "$run_idx" 0 100 nothing
    done
done

# PDE problems
for pname in DuffingProblem DiffusionProblem10; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_warpedgp-maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --time=3-00:00:00 --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-maxvar "$run_idx" 0 100 nothing
    done
done

echo "Done. 35 jobs submitted."
