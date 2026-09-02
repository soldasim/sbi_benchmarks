#!/bin/bash
# Submit EIV runs for all 24 sharp opt-function problems (2D, 5 runs each).
# Run from ~/repos/bosip_benchmarks after setup_sharp_opt_problems.sh has completed.

cd ~/repos/bosip_benchmarks

problems=(
    "RosenbrockProblem2_sharp"
    "StyblinskiTangProblem2_sharp"
    "MichalewiczProblem2_sharp"
    "AckleyProblem2_sharp"
    "AlpineProblem2_sharp"
    "ExpandedSchafferF6Problem2_sharp"
    "ExpandedZakharovProblem2_sharp"
    "GriewankProblem2_sharp"
    "RastriginProblem2_sharp"
    "SalomonProblem2_sharp"
    "SchwefelProblem2_sharp"
    "SphereProblem2_sharp"
    "BealeProxyProblem_sharp"
    "BoothProblem_sharp"
    "CrossInTrayProblem_sharp"
    "DropWaveProblem_sharp"
    "EasomProblem_sharp"
    "GoldsteinPriceProxyProblem_sharp"
    "HimmelblauProblem_sharp"
    "HolderTableProblem_sharp"
    "LeviN13Problem_sharp"
    "MatyasProblem_sharp"
    "SchafferN2Problem_sharp"
    "ThreeHumpCamelProblem_sharp"
)

job_ids=()
count=0
for pname in "${problems[@]}"; do
    for run_idx in 1 2 3 4 5; do
        job_name="${pname}_eiv_${run_idx}"
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 200 nothing)
        job_ids+=("$jid")
        count=$((count + 1))
    done
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
