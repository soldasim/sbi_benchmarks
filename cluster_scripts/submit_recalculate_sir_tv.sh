#!/bin/bash
# Recompute TV metrics (with posterior normalization) for old SIR runs.
# Writes *_TVmetric_norm.jld2 alongside existing *_TVmetric.jld2 — originals are NOT modified.
#
# Workflow:
#   1. Precompute posterior grid for SIRProblem (needed by get_metric).
#   2. Submit 80 score-calculation jobs dependent on the precompute finishing.
#
# Run from: ~/repos/bosip_benchmarks

cd ~/repos/bosip_benchmarks

chmod +x cluster_scripts/run_score.sh

metric="TVMetric"
estimator="log_posterior_mean"
suffix="_norm"

# Step 1 — precompute grid for SIRProblem
echo "Submitting SIRProblem grid precompute job..."
precompute_jid=$(sbatch --parsable -p cpufast --time=04:00:00 --mem=12G \
    --job-name="SIRProblem_precompute" \
    cluster_scripts/precompute_grid.sh SIRProblem)
echo "  Precompute job ID: $precompute_jid"

# Step 2 — 60 SIRProblem jobs (standard x20, eiv x20, nongp x20)
echo "Submitting SIRProblem score jobs (dependent on precompute)..."
for run_name in standard eiv nongp; do
    for run_idx in $(seq 1 20); do
        job_name="${metric}_SIRProblem_${run_name}_${run_idx}"
        jid=$(sbatch --parsable --dependency=afterok:${precompute_jid} \
            -p cpufast --time=04:00:00 --mem=16G \
            --job-name="$job_name" \
            cluster_scripts/run_score.sh SIRProblem "$run_name" "$run_idx" "$metric" "$estimator" "$suffix")
        echo "  Submitted $job_name -> job $jid"
    done
done

# Step 3 — 20 ProxySIRProblem/standard jobs (grid already exists, no dependency needed)
echo "Submitting ProxySIRProblem/standard score jobs..."
for run_idx in $(seq 1 20); do
    job_name="${metric}_ProxySIRProblem_standard_${run_idx}"
    jid=$(sbatch --parsable \
        -p cpufast --time=04:00:00 --mem=16G \
        --job-name="$job_name" \
        cluster_scripts/run_score.sh ProxySIRProblem standard "$run_idx" "$metric" "$estimator" "$suffix")
    echo "  Submitted $job_name -> job $jid"
done

echo "Done. 1 precompute + 80 score jobs submitted."
