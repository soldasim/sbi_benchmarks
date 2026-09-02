#!/bin/bash
# Submit 5D BIP benchmark experiments: DuffingProblem5 and DiffusionProblem5D.
# 5 runs × 4 configs (standard, maxvar, eiv, immd) × 2 problems = 40 jobs.
# Data stored in data-bosip-norm/<ProblemName>/.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# standard and maxvar: no integral acquisition — fast, cpu (1 day) is sufficient
for pname in DuffingProblem5 DiffusionProblem5D; do
    for run_name in standard maxvar; do
        for run_idx in 1 2 3 4 5; do
            job_name="${pname}_${run_name}_${run_idx}"
            echo "Submitting $job_name ..."
            sbatch -p cpu --mem=12G --job-name="$job_name" \
                cluster_scripts/run.sh "$pname" "$run_name" "$run_idx" 0 100 nothing
        done
    done
done

# eiv and immd: integral acquisition with 2000 x-samples — slow at 5D, use cpulong
for pname in DuffingProblem5 DiffusionProblem5D; do
    for run_name in eiv immd; do
        for run_idx in 1 2 3 4 5; do
            job_name="${pname}_${run_name}_${run_idx}"
            echo "Submitting $job_name ..."
            sbatch -p cpulong --mem=12G --job-name="$job_name" \
                cluster_scripts/run.sh "$pname" "$run_name" "$run_idx" 0 100 nothing
        done
    done
done

echo "Done. 40 jobs submitted."
