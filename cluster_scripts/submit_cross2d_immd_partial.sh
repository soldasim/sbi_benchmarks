#!/bin/bash
# Partial submission: submit only the first MAX_PROBLEMS from the Group B IMMD bulk list.
# Usage: bash cluster_scripts/submit_cross2d_immd_partial.sh <max_problems>
# Example: bash cluster_scripts/submit_cross2d_immd_partial.sh 15
# Submits max_problems * 20 jobs.

MAX_PROBLEMS=${1:?Usage: $0 <max_problems>}

cd ~/repos/bosip_benchmarks

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

job_ids=()
count=0
prob_count=0

echo "Submitting first ${MAX_PROBLEMS} problems x 20 runs = $((MAX_PROBLEMS * 20)) jobs"
echo "NOTE: BoothProblem_cross immd_1 (test job file) will be overwritten by run 1."
echo ""

for pname in "${probs[@]}"; do
    [ $prob_count -ge $MAX_PROBLEMS ] && break
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
    prob_count=$((prob_count + 1))
done

echo ""
echo "Submitted $count jobs across $prob_count problems."
echo "JOB_IDS (first/last): ${job_ids[0]} ... ${job_ids[-1]}"
