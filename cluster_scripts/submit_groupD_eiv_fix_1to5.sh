#!/bin/bash
# Fix for Group D EIV runs 21-25 that failed (start_21-25 don't exist).
# Resubmit using run indices 1-5 instead (no existing EIV data for Group D at indices 1-5).

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

for pname in \
    "RosenbrockProblem5_cross" \
    "StyblinskiTangProblem5_cross" \
    "MichalewiczProblem5_cross" \
    "SphereProblem5_cross"; do
    for run_idx in $(seq 1 5); do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) || echo "  -> FAILED (QOS limit)"
    done
done

echo "Done. $count jobs submitted successfully."
echo "JOB_IDS: ${job_ids[*]}"
