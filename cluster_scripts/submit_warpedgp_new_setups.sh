#!/bin/bash
# Submit new WarpedGP setups (fixed GP mean=0, amplitude=1, Affine warping layer):
#   warpedgp-yja-maxvar  : YJ + Affine
#   warpedgp-yjsa-maxvar : YJ + SinhArcsinh + Affine
# Run on 8 BIP benchmark problems + Beale + GoldsteinPrice, 1 run each.
# Data stored in data-warpedgp2/<ProblemName>/.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# 8 BIP benchmark problems
for pname in ABProblem SimpleProblem BananaProblem BimodalProblem SIRProblem ProxySIRProblem DuffingProblem DiffusionProblem10; do
    for rname in warpedgp-yja-maxvar warpedgp-yjsa-maxvar; do
        job_name="${pname}_${rname}_1"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" "$rname" 1 0 100 nothing
    done
done

# Optimization-function problems (no proxy, raw output)
for pname in BealeProblem GoldsteinPriceProblem; do
    for rname in warpedgp-yja-maxvar warpedgp-yjsa-maxvar; do
        job_name="${pname}_${rname}_1"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" "$rname" 1 0 100 nothing
    done
done

echo "Done. 20 jobs submitted."
