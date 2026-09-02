#!/bin/bash
# Rerun warpedgp-yja-maxvar for Group B (24 2D cross-polytope opt-function problems)
# after the predictive_samples fix (2026-07-23), mirroring the Group A rerun.
# Old data for all 30 Group B/C/D problems already archived into
# data-warpedgp2_archive_pre-samplesfix/ (swept up as part of Group A's full-directory
# archive move). Fresh data-warpedgp2/<Problem>_cross/ dirs will be created by run.sh.
#
# RosenbrockProblem2_cross run 1 was already submitted as the debug/test job (11202555,
# verified clean) — skip run 1 for it here to avoid two concurrent jobs writing the same
# output file (see debug-job-hygiene incident in reference_rci_cluster.md).
#
# Partition: cpu (24h) — 200 iters should complete well within 24h at d=2.
# Data stored in data-warpedgp2/<ProblemName>_cross/.

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
for pname in "${problems[@]}"; do
    start_idx=1
    if [ "$pname" == "RosenbrockProblem2_cross" ]; then
        start_idx=2  # run 1 already submitted as test job 11202555
    fi
    for run_idx in $(seq $start_idx 20); do
        job_name="${pname}_wgp_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
