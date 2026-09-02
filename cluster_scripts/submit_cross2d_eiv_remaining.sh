#!/bin/bash
# Submit remaining EIV jobs that failed due to QOS limit in the first EIV batch.
# Missing: BealeProblem_cross runs 10-20, and problems 14-24 runs 6-20,
# plus fresh reruns for bad runs (Booth 21-23, Himmelblau 21).
# Run from ~/repos/bosip_benchmarks when queue has room.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

# BealeProblem_cross: runs 10-20 (runs 6-9 already submitted)
for run_idx in $(seq 10 20); do
    job_name="BealeProblem_cross_eiv_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "BealeProblem_cross" eiv "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit?)"
done

# Problems 14-24: runs 6-20
for pname in \
    "BoothProblem_cross" \
    "CrossInTrayProblem_cross" \
    "DropWaveProblem_cross" \
    "EasomProblem_cross" \
    "GoldsteinPriceProblem_cross" \
    "HimmelblauProblem_cross" \
    "HolderTableProblem_cross" \
    "LeviN13Problem_cross" \
    "MatyasProblem_cross" \
    "SchafferN2Problem_cross" \
    "ThreeHumpCamelProblem_cross"; do
    for run_idx in $(seq 6 20); do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit?)"
    done
done

# Fresh reruns for bad runs
for run_idx in 21 22 23; do
    job_name="BoothProblem_cross_eiv_${run_idx}"
    echo "Submitting $job_name (fresh rerun for bad original run) ..."
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "BoothProblem_cross" eiv "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit?)"
done

job_name="HimmelblauProblem_cross_eiv_21"
echo "Submitting $job_name (fresh rerun for bad original run 3) ..."
jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "HimmelblauProblem_cross" eiv 21 0 200 nothing)
[ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit?)"

echo "Done. $count jobs submitted successfully."
echo "JOB_IDS: ${job_ids[*]}"
