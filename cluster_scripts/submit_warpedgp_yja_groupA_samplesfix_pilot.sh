#!/bin/bash
# Preliminary rerun of warpedgp-yja-maxvar after the predictive_samples fix
# (BOSIP.jl now routes log_likelihood_mean/variance through predictive_samples
# for SampledPredictive models like WarpedGaussianProcess, instead of the old
# moment-matched Ey/Ey2 quadrature collapse).
#
# 5 runs x 7 original BIP problems, 100 iters each.
# Output: data-warpedgp2/<ProblemName>/ (fresh dir; old buggy data archived to
# data-warpedgp2_archive_pre-samplesfix/).
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

for pname in ABProblem SimpleProblem BananaProblem BimodalProblem SIRProblem DuffingProblem DiffusionProblem10; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_warpedgp-yja-maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-yja-maxvar "$run_idx" 0 100 nothing
    done
done

echo "Done. 35 jobs submitted."
