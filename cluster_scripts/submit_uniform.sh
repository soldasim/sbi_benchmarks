#!/bin/bash
# Submit 45 uniform benchmark jobs (5 runs x 9 problem/dim combos)
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

# MeanGauss dims 1-6
for d in 1 2 3 4 5 6; do
    pname="MeanGauss${d}"
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_uniform_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" uniform "$run_idx" 0 1000 nothing
    done
done

# MultidimProblem(ABProblem(), 1/2/3) -> scaleup 1=2D, 2=4D, 3=6D
for scaleup in 1 2 3; do
    pname="MultidimProblem{ABProblem}${scaleup}"
    for run_idx in 1 2 3 4 5; do
        job_name="MultidimAB${scaleup}_uniform_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" uniform "$run_idx" 0 1000 nothing
    done
done

echo "Done. Total 45 jobs submitted."

# New 1D analytical problems (SquareProblem, SineProblem, CubicProblem), dims 1-6
for base in SquareProblem SineProblem CubicProblem; do
    for scaleup in 1 2 3 4 5 6; do
        pname="MultidimProblem{}"
        for run_idx in 1 2 3 4 5; do
            job_name="_uniform_"
            echo "Submitting $job_name ..."
            sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" uniform "$run_idx" 0 1000 nothing
        done
    done
done
