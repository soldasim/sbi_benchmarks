#!/bin/bash
# Extend the samplesfix-verified warpedgp-yja-maxvar rerun (see
# submit_warpedgp_yja_groupA_samplesfix_pilot.sh, runs 1-5, verified clean +
# NaN-free 2026-07-23) to the full 20-run target: runs 6-20.
#
# 15 runs x 7 original Group A BIP problems = 105 jobs, 100 iters each.
# Output: data-warpedgp2/<ProblemName>/ (same fresh post-fix dir as runs 1-5).
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

for pname in ABProblem SimpleProblem BananaProblem BimodalProblem SIRProblem DuffingProblem DiffusionProblem10; do
    for run_idx in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        job_name="${pname}_warpedgp-yja-maxvar_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-yja-maxvar "$run_idx" 0 100 nothing
    done
done

echo "Done. 105 jobs submitted."
