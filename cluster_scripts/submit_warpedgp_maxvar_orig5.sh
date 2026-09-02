#!/bin/bash
# Submit WarpedGaussianProcess + LogMaxVar on the 5 original benchmark problems
# that don't yet have WarpedGP data (AB and ProxySIR already done in the pilot).
# 5 runs × 5 problems = 25 jobs, 100 iters each.
# Data stored in data-warpedgp/<ProblemName>/.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# Analytical 2D problems
for pname in SimpleProblem BananaProblem BimodalProblem; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_warpedgp-maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-maxvar "$run_idx" 0 100 nothing
    done
done

# PDE problems
for pname in DuffingProblem DiffusionProblem10; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_warpedgp-maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-maxvar "$run_idx" 0 100 nothing
    done
done

echo "Done. 25 jobs submitted."
