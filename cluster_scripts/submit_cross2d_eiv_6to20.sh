#!/bin/bash
# Submit EIV runs 6–20 for all 24 2D cross-polytope opt-function problems,
# plus fresh-index reruns for known bad runs:
#   BoothProblem_cross eiv runs 1,2,5 (57–96 iters, bad) → fresh runs at indices 21,22,23
#   HimmelblauProblem_cross eiv run 3 (101 iters, bad)   → fresh run at index 21
# Includes BealeProblem_cross and GoldsteinPriceProblem_cross (new, non-proxy).
# Run from ~/repos/bosip_benchmarks after setup is complete.
#
# Partition: cpulong (72h) — cross EIV at d=2 with 4D output takes ~7 min/iter;
# 200 iters ≈ 23h, which exceeds the cpu 24h limit.

cd ~/repos/bosip_benchmarks

problems=(
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
    "BealeProblem_cross"
    "BoothProblem_cross"
    "CrossInTrayProblem_cross"
    "DropWaveProblem_cross"
    "EasomProblem_cross"
    "GoldsteinPriceProblem_cross"
    "HimmelblauProblem_cross"
    "HolderTableProblem_cross"
    "LeviN13Problem_cross"
    "MatyasProblem_cross"
    "SchafferN2Problem_cross"
    "ThreeHumpCamelProblem_cross"
)

job_ids=()
count=0

# Runs 6–20 for all 24 problems
for pname in "${problems[@]}"; do
    for run_idx in $(seq 6 20); do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

# Fresh reruns for bad BoothProblem_cross eiv runs (original 1,2,5 bad at 57–96 iters)
for run_idx in 21 22 23; do
    job_name="BoothProblem_cross_eiv_${run_idx}"
    echo "Submitting $job_name (fresh rerun for bad original run) ..."
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "BoothProblem_cross" eiv "$run_idx" 0 200 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
done

# Fresh rerun for bad HimmelblauProblem_cross eiv run 3 (101 iters, bad)
job_name="HimmelblauProblem_cross_eiv_21"
echo "Submitting $job_name (fresh rerun for bad original run 3) ..."
jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "HimmelblauProblem_cross" eiv 21 0 200 nothing)
job_ids+=("$jid")
count=$((count + 1))

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
