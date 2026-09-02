#!/bin/bash
# Submit tvnorm jobs not yet complete and not already in the queue.
# Safe to run repeatedly — skips completed output files and already-queued jobs.

cd ~/repos/bosip_benchmarks

problems=("ABProblem" "SimpleProblem" "BananaProblem" "BimodalProblem"
          "SIRProblem" "DuffingProblem" "DiffusionProblem10" "ProxySIRProblem")
run_names=("standard" "eiv" "eiig" "nongp")

# Snapshot current queue job names once
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

            # Already computed — skip
            if [ -f "$out_file" ]; then
                ((skipped_done++))
                continue
            fi

            # Already in queue — skip
            if echo "$queued" | grep -qx "$job_name"; then
                ((skipped_queued++))
                continue
            fi

            # Try to submit
            result=$(sbatch -p cpufast --mem=8G --time=2:00:00 \
                --job-name="$job_name" \
                cluster_scripts/run_recompute.sh "$pname" "$rname" "$run_idx" 2>&1)
            if echo "$result" | grep -q "error"; then
                ((errors++))
                # Stop trying once we hit the limit
                if echo "$result" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
                    echo "QOS limit hit. Submitted $submitted this run."
                    echo "Done: $skipped_done  Queued: $skipped_queued  New: $submitted  Errors: $errors"
                    exit 1
                fi
            else
                ((submitted++))
            fi
        done
    done
done

echo "Done: $skipped_done  Queued: $skipped_queued  New: $submitted  Errors: $errors"
