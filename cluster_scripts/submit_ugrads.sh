#!/bin/bash
# Submit 45 uniform-grads benchmark jobs (MeanGauss d=1-6, MultidimAB scaleup=1-3, 5 runs each)

cd ~/repos/bosip_benchmarks

for d in 1 2 3 4 5 6; do
    pname="MeanGauss${d}"
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_ugrads_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" uniform-grads "$run_idx" 0 1000 nothing
    done
done

for scaleup in 1 2 3; do
    pname="MultidimProblem{ABProblem}${scaleup}"
    for run_idx in 1 2 3 4 5; do
        job_name="MultidimAB${scaleup}_ugrads_${run_idx}"
        echo "Submitting $job_name ..."
        sbatch -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" uniform-grads "$run_idx" 0 1000 nothing
    done
done

echo "Done. 45 jobs submitted."
