#!/bin/bash
# Submit EIV runs 6-20 for Group C (2 HD BIP problems).
# DuffingProblem5 + DiffusionProblem5D, runs 6-20, run_name=eiv, cpulong, 200 iters.
# Submit after Group C EIV runs 1-5 (jobs 11101431-11101445) complete (~2026-07-03 00:20).
# 2 problems x 15 runs = 30 jobs.

cd ~/repos/bosip_benchmarks

problems=("DuffingProblem5" "DiffusionProblem5D")

job_ids=()
count=0

for pname in "${problems[@]}"; do
    for run_idx in $(seq 6 20); do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="$job_name" \
            cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
