#!/bin/bash
# Bulk submission: Group B IMMD for all 24 cross 2D problems.
# Target: 100 iters, cpu partition (~5-6h per run, safely within 24h limit).
# Based on timing from BoothProblem_cross immd test: ~3 min/iter early, slowing to ~4 min/iter.
# Generated 2026-07-03. Run AFTER test job 11112958 confirms clean completion.
#
# Usage: check current queue count before running:
#   squeue -u soldasim --noheader | wc -l
# Only run if (current_count + 480) <= 500 QOS limit.
# If close to limit, submit in batches or wait for jobs to clear.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

submit() {
    local pname="$1" run_idx="$2"
    local job_name="${pname}_immd_${run_idx}"
    jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" \
          cluster_scripts/run.sh "$pname" immd "$run_idx" 0 100 nothing)
    if [ -n "$jid" ]; then
        job_ids+=("$jid")
        count=$((count + 1))
        echo "  $job_name -> $jid"
    else
        echo "FAILED: $job_name"
    fi
}

probs=(
    "RosenbrockProblem2_cross"
    "StyblinskiTangProblem2_cross"
    "MichalewiczProblem2_cross"
    "AckleyProblem2_cross"
    "AlpineProblem2_cross"
    "ExpandedSchafferF6Problem2_cross"
    "ExpandedZakharovProblem2_cross"
    "GriewankProblem2_cross"
    "RastriginProblem2_cross"
    "SalomonProblem2_cross"
    "SchwefelProblem2_cross"
    "SphereProblem2_cross"
    "BoothProblem_cross"
    "CrossInTrayProblem_cross"
    "DropWaveProblem_cross"
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

echo "Submitting Group B IMMD: 24 problems × 20 runs = 480 jobs"
echo "NOTE: BoothProblem_cross immd_1 is the test job (11112958). Run 1 will be overwritten."
echo ""

for pname in "${probs[@]}"; do
    echo "--- $pname ---"
    for run_idx in $(seq 1 20); do
        submit "$pname" "$run_idx"
    done
done

echo ""
echo "Submitted $count jobs total."
echo "JOB_IDS (first/last): ${job_ids[0]} ... ${job_ids[-1]}"
