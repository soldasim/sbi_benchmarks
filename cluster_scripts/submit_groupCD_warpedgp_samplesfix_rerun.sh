#!/bin/bash
# Rerun warpedgp-yja-maxvar for Groups C (2 HD BIP) and D (4 HD cross opt) after the
# predictive_samples fix (2026-07-23), mirroring the Group A rerun.
# Old data for all 30 Group B/C/D problems already archived into
# data-warpedgp2_archive_pre-samplesfix/ (swept up as part of Group A's full-directory
# archive move). Fresh data-warpedgp2/<Problem>/ dirs will be created by run.sh.
#
# DuffingProblem5 run 1 (test job 11202556) and RosenbrockProblem5_cross run 1 (test job
# 11202557) already submitted and verified clean — skip run 1 for both here to avoid two
# concurrent jobs writing the same output file (see debug-job-hygiene incident in
# reference_rci_cluster.md).
#
# Group C: DuffingProblem5 + DiffusionProblem5D, cpulong, 200 iters.
# Group D: 4 HD cross opt problems, cpulong, 200 iters.
# Data stored in data-warpedgp2/<ProblemName>/.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

# Group C — DuffingProblem5 (run 1 already submitted as test job 11202556)
for run_idx in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    job_name="DuffingProblem5_warpedgp-yja_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
        cluster_scripts/run.sh DuffingProblem5 warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
done

# Group C — DiffusionProblem5D runs 1-20
for run_idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    job_name="DiffusionProblem5D_warpedgp-yja_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
        cluster_scripts/run.sh DiffusionProblem5D warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
    job_ids+=("$jid")
    count=$((count + 1))
done

# Group D — 4 HD cross opt problems, runs 1-20 (RosenbrockProblem5_cross run 1 already
# submitted as test job 11202557)
group_d_problems=(
    "RosenbrockProblem5_cross"
    "StyblinskiTangProblem5_cross"
    "MichalewiczProblem5_cross"
    "SphereProblem5_cross"
)
for pname in "${group_d_problems[@]}"; do
    start_idx=1
    if [ "$pname" == "RosenbrockProblem5_cross" ]; then
        start_idx=2
    fi
    for run_idx in $(seq $start_idx 20); do
        job_name="${pname}_warpedgp-yja_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" warpedgp-yja-maxvar "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
