#!/bin/bash
# Submit nongp (MaxVar acq + NonstatGP surrogate) runs 1-20 for all 24 Group B
# 2D cross-polytope opt-function problems. None of these problems have any
# nongp data yet (verified 2026-07-24). Uses the paper's non-proxy Beale/
# GoldsteinPrice variants (BealeProblem_cross / GoldsteinPriceProblem_cross),
# matching the "non-proxy everywhere" design rationale for Groups A-D.
#
# cpu partition (1 day), 100 iters. Run from ~/repos/bosip_benchmarks.

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

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)

job_ids=()
count=0
skipped=0
for pname in "${problems[@]}"; do
    for run_idx in $(seq 1 20); do
        job_name="${pname}_nongp_${run_idx}"
        if echo "$queued" | grep -qx "$job_name"; then
            echo "Skipping $job_name (already queued/running, e.g. a debug-test job)"
            skipped=$((skipped + 1))
            continue
        fi
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" nongp "$run_idx" 0 100 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted, $skipped skipped (already queued)."
echo "JOB_IDS: ${job_ids[*]}"
