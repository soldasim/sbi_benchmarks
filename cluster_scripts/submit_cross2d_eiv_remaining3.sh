#!/bin/bash
# Submit remaining 95 EIV jobs (cpulong) that couldn't fit due to QOS partition limit.
# Missing: GP run 20; Himmelblau/HolderTable/LeviN13/Matyas/SchafferN2/ThreeHumpCamel runs 6-20;
#          Booth fresh reruns 21-23; Himmelblau fresh rerun 21.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

# GoldsteinPriceProblem_cross: only run 20 missing (runs 6-19 already submitted)
job_name="GoldsteinPriceProblem_cross_eiv_20"
echo "Submitting $job_name ..."
jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "GoldsteinPriceProblem_cross" eiv 20 0 200 nothing)
[ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"

# Problems with all runs 6-20 missing
for pname in \
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
        [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"
    done
done

# Fresh reruns for bad runs
for run_idx in 21 22 23; do
    job_name="BoothProblem_cross_eiv_${run_idx}"
    echo "Submitting $job_name (fresh rerun for bad original run) ..."
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "BoothProblem_cross" eiv "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"
done

job_name="HimmelblauProblem_cross_eiv_21"
echo "Submitting $job_name (fresh rerun for bad original run 3) ..."
jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "HimmelblauProblem_cross" eiv 21 0 200 nothing)
[ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"

echo "Done. $count jobs submitted successfully."
echo "JOB_IDS: ${job_ids[*]}"
