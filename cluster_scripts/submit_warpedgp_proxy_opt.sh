#!/bin/bash
# Submit WarpedGaussianProcess (maxvar + eiv) on opt-function problems where we
# used proxy variables (Beale and GoldsteinPrice), running on the NON-proxy variants.
# Goal: test whether WarpedGP can handle the raw functions without a hand-crafted proxy.
# 5 runs × 2 acquisitions × 2 problems = 20 jobs, 100 iters each.
# Data stored in data-warpedgp/<ProblemName>/.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

for pname in BealeProblem GoldsteinPriceProblem; do
    for rname in warpedgp-maxvar warpedgp-eiv; do
        for run_idx in 1 2 3 4 5; do
            job_name="${pname}_${rname}_${run_idx}"
            echo "Submitting $job_name ..."
            sbatch -p cpulong --mem=12G --job-name="$job_name" \
                cluster_scripts/run.sh "$pname" "$rname" "$run_idx" 0 100 nothing
        done
    done
done

echo "Done. 20 jobs submitted."
