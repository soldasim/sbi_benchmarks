#!/bin/bash
# Orchestrate ProxySIRProblem extra experiments:
# 1. Precompute the posterior/simulator grids (needed for TVMetric)
# 2. Submit 15 experiment jobs (eiv x5, eiig x5, nongp x5) depending on precompute finishing.
# Run from ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

chmod +x cluster_scripts/precompute_grid.sh

echo "Submitting precompute job for ProxySIRProblem ..."
precompute_jid=$(sbatch --parsable -p cpulong --time=3-00:00:00 --mem=12G \
    --job-name="ProxySIRProblem_precompute" \
    cluster_scripts/precompute_grid.sh ProxySIRProblem)
echo "  Precompute job ID: $precompute_jid"

echo "Submitting 15 experiment jobs (dependent on precompute) ..."
pname="ProxySIRProblem"
for run_name in eiv eiig nongp; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_${run_name}_${run_idx}"
        jid=$(sbatch --parsable --dependency=afterok:$precompute_jid \
            -p cpulong --time=3-00:00:00 --mem=12G \
            --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" "$run_name" "$run_idx" 0 100 nothing)
        echo "  Submitted $job_name -> job $jid"
    done
done

echo "Done. Precompute job $precompute_jid + 15 experiment jobs submitted."
