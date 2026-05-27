#!/bin/bash
# Submit 90 uniform benchmark jobs for new analytical problems
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

for base in SquareProblem SineProblem CubicProblem; do
    for scaleup in 1 2 3 4 5 6; do
        pname="MultidimProblem{${base}}${scaleup}"
        for run_idx in 1 2 3 4 5; do
            shortname="${base%%Problem}"
            job_name="${shortname}_d${scaleup}_uniform_${run_idx}"
            echo "Submitting $job_name ..."
            sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" uniform "$run_idx" 0 1000 nothing
        done
    done
done

echo "Done. 90 jobs submitted."
