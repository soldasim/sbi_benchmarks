#!/bin/bash
# Submit WarpedGP (maxvar + eiv) on ABProblem, SIRProblem, ProxySIRProblem.
# 5 runs × 2 acquisitions × 3 problems = 30 jobs, 100 iters each.
# Output: data-warpedgp/<ProblemName>/
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# ABProblem — fast analytical 2D, cpu partition sufficient
for rname in warpedgp-maxvar warpedgp-eiv; do
    for run_idx in 1 2 3 4 5; do
        job_name="ABProblem_${rname}_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpu --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh ABProblem "$rname" "$run_idx" 0 100 nothing
    done
done

# SIRProblem — physical ODE simulator, cpulong
for rname in warpedgp-maxvar warpedgp-eiv; do
    for run_idx in 1 2 3 4 5; do
        job_name="SIRProblem_${rname}_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh SIRProblem "$rname" "$run_idx" 0 100 nothing
    done
done

# ProxySIRProblem — physical ODE simulator with proxy, cpulong
for rname in warpedgp-maxvar warpedgp-eiv; do
    for run_idx in 1 2 3 4 5; do
        job_name="ProxySIRProblem_${rname}_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh ProxySIRProblem "$rname" "$run_idx" 0 100 nothing
    done
done

echo "Done. 30 jobs submitted."
