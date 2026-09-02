#!/bin/bash
# Re-submit tvnorm jobs that are neither done nor currently queued.
# Uses the `cpu` partition (24h limit) to handle slow nongp/eiig runs.
# Safe to run repeatedly — skips completed output files and already-queued jobs.

cd ~/repos/bosip_benchmarks

problems=("ABProblem" "SimpleProblem" "BananaProblem" "BimodalProblem" \
          "SIRProblem" "DuffingProblem" "DiffusionProblem10" "ProxySIRProblem")
run_names=("standard" "eiv" "eiig" "nongp")

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)

submitted=0
skipped_done=0
skipped_queued=0
errors=0

for pname in "${problems[@]}"; do
    for rname in "${run_names[@]}"; do
        for run_idx in $(seq 1 20); do
            job_name="${pname}_${rname}_tvnorm_${run_idx}"
            out_file="data-bosip-norm/${pname}/${rname}_${run_idx}_TVmetric.jld2"

            src_file="data-bosip/${pname}/${rname}_${run_idx}_data.jld2"
            [ ! -f "$src_file" ] && continue  # no source data, nothing to compute

            if [ -f "$out_file" ]; then
                ((skipped_done++))
                continue
            fi

            if echo "$queued" | grep -qx "$job_name"; then
                ((skipped_queued++))
                continue
            fi

            result=$(sbatch -p cpu --mem=8G \
                --job-name="$job_name" \
                cluster_scripts/run_recompute.sh "$pname" "$rname" "$run_idx" 2>&1)
            if echo "$result" | grep -q "error"; then
                echo "ERROR submitting $job_name: $result"
                ((errors++))
            else
                ((submitted++))
            fi
        done
    done
done

echo "Done: $skipped_done  Queued: $skipped_queued  New: $submitted  Errors: $errors"
