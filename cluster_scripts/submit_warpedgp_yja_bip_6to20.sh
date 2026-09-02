#!/bin/bash
# Submit warpedgp-yja-maxvar runs 6–20 for the 7 original BIP problems.
# Runs 1–5 already exist in data-warpedgp2/.
# Data stored in data-warpedgp2/<ProblemName>/.
# 15 runs × 7 problems = 105 jobs on cpulong.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

for pname in ABProblem SimpleProblem BananaProblem BimodalProblem SIRProblem DuffingProblem DiffusionProblem10; do
    for idx in $(seq 6 20); do
        job_name="${pname}_warpedgp-yja-maxvar_${idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-yja-maxvar $idx 0 100 nothing
    done
done

echo "Done. 105 jobs submitted."
