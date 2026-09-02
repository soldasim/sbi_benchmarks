#!/bin/bash
# Submit remaining 45 EIV jobs (cpulong).
# MatyasProblem_cross run 9 already submitted as 11106644.
# Missing: Matyas 10-20; SchafferN2/ThreeHumpCamel 6-20; Booth reruns 21-23; Himmelblau rerun 21.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

# MatyasProblem_cross: runs 10-20 (run 9 already submitted)
for run_idx in $(seq 10 20); do
    job_name="MatyasProblem_cross_eiv_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "MatyasProblem_cross" eiv "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"
done

# Problems with all runs 6-20 missing
for pname in "SchafferN2Problem_cross" "ThreeHumpCamelProblem_cross"; do
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
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "BoothProblem_cross" eiv "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED"
done

jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="HimmelblauProblem_cross_eiv_21" cluster_scripts/run.sh "HimmelblauProblem_cross" eiv 21 0 200 nothing)
[ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED"

echo "Done. $count jobs submitted successfully."
echo "JOB_IDS: ${job_ids[*]}"
