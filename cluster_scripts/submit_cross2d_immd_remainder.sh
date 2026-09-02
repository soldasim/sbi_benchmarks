#!/bin/bash
# Submit Group B IMMD remainder: problems 16-24 (9 problems × 20 runs = 180 jobs).
# Run ONLY after batch 1 (problems 1-15, IDs 11114076-11114375) has been submitted.
# Only run when queue count <= 270 (need 180 budget).

cd ~/repos/bosip_benchmarks

probs=(
    "EasomProblem_cross"
    "HimmelblauProblem_cross"
    "HolderTableProblem_cross"
    "LeviN13Problem_cross"
    "MatyasProblem_cross"
    "SchafferN2Problem_cross"
    "ThreeHumpCamelProblem_cross"
    "BealeProblem_cross"
    "GoldsteinPriceProblem_cross"
)

job_ids=()
count=0

echo "Submitting Group B IMMD remainder: 9 problems x 20 runs = 180 jobs"
echo ""

for pname in "${probs[@]}"; do
    echo "--- $pname ---"
    for run_idx in $(seq 1 20); do
        job_name="${pname}_immd_${run_idx}"
        jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" \
              cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing)
        if [ -n "$jid" ]; then
            job_ids+=("$jid")
            count=$((count + 1))
            echo "  $job_name -> $jid"
        else
            echo "FAILED: $job_name"
        fi
    done
done

echo ""
echo "Submitted $count jobs total."
echo "JOB_IDS (first/last): ${job_ids[0]} ... ${job_ids[-1]}"
