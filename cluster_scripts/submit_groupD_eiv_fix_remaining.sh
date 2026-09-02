#!/bin/bash
# Remaining Group D EIV fix jobs (runs 1-5 replacing failed 21-25):
#   RosenbrockProblem5_cross: runs 3-5 (runs 1-2 submitted as 11106621-11106622)
#   StyblinskiTangProblem5_cross: runs 1-5
#   MichalewiczProblem5_cross: runs 1-5
#   SphereProblem5_cross: runs 1-5

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

for run_idx in $(seq 3 5); do
    job_name="RosenbrockProblem5_cross_eiv_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "RosenbrockProblem5_cross" eiv "$run_idx" 0 200 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"
done

for pname in "StyblinskiTangProblem5_cross" "MichalewiczProblem5_cross" "SphereProblem5_cross"; do
    for run_idx in $(seq 1 5); do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"
    done
done

echo "Done. $count jobs submitted successfully."
echo "JOB_IDS: ${job_ids[*]}"
