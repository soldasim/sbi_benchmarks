#!/bin/bash
# Recompute TV metric scores for all original data-bosip benchmark runs.
# Reads observation data from data-bosip/, writes corrected TVmetric files
# to data-bosip-norm/.
#
# 8 problems x 4 run types x 20 runs = up to 640 jobs.
# Jobs for non-existent source files exit immediately without error.

cd ~/repos/bosip_benchmarks

problems=("ABProblem" "SimpleProblem" "BananaProblem" "BimodalProblem" \
          "SIRProblem" "DuffingProblem" "DiffusionProblem10" "ProxySIRProblem")
run_names=("standard" "eiv" "eiig" "nongp")

for pname in "${problems[@]}"; do
    for rname in "${run_names[@]}"; do
        for run_idx in $(seq 1 20); do
            job_name="${pname}_${rname}_tvnorm_${run_idx}"
            sbatch -p cpufast --mem=8G --time=2:00:00 \
                --job-name="$job_name" \
                cluster_scripts/run_recompute.sh "$pname" "$rname" "$run_idx"
        done
    done
done

echo "Done. Up to 640 jobs submitted (jobs with no source data exit immediately)."
