#!/bin/bash
# Submit maxvar benchmark jobs for analytical problems (Square/Sine/Cubic, d=1-6, 5 runs each)
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

for base in SquareProblem SineProblem CubicProblem; do
    for scaleup in 1 2 3 4 5 6; do
        pname="MultidimProblem{${base}}${scaleup}"
        for run_idx in 1 2 3 4 5; do
            shortname="${base%%Problem}"
            job_name="${shortname}_d${scaleup}_maxvar_${run_idx}"
            echo "Submitting $job_name ..."
            sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" maxvar "$run_idx" 0 1000 nothing
        done
    done
done

echo "Done. 90 jobs submitted."
