#!/bin/bash
# Submit ProxySIRProblem benchmark runs with NonstationaryGP model (LogMaxVar acquisition).
# Runs 1-5, 100 iters each, matching the original standard run setup.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

pname="ProxySIRProblem"

for run_idx in 1 2 3 4 5; do
    job_name="${pname}_nongp_${run_idx}"
    echo "Submitting $job_name ..."
    sbatch -p cpulong --time=3-00:00:00 --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" nongp "$run_idx" 0 100 nothing
done

echo "Done. 5 jobs submitted."
