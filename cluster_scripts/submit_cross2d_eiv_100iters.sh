#!/bin/bash
# Resubmission of Group B EIV runs with 100-iter target on cpu partition.
# Submits only runs that are missing or have < 100 iters on disk (per audit 2026-07-01).
# 4 problems already complete (Michalewicz2, Ackley2, Alpine2, ExpandedSchafferF6_2) — not listed.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

submit() {
    local pname="$1" run_idx="$2"
    local job_name="${pname}_cross_eiv_${run_idx}"
    jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" \
          cluster_scripts/run.sh "${pname}_cross" eiv "$run_idx" 0 100 nothing)
    if [ -n "$jid" ]; then
        job_ids+=("$jid")
        count=$((count + 1))
    else
        echo "FAILED: $job_name"
    fi
}

# --- RosenbrockProblem2: bad runs (< 100 iters on disk) ---
for run_idx in 7 10 11 12 13 14 15 16 17 18 19 20; do
    submit "RosenbrockProblem2" "$run_idx"
done

# --- StyblinskiTangProblem2: bad runs ---
for run_idx in 7 17 20; do
    submit "StyblinskiTangProblem2" "$run_idx"
done

# --- ExpandedZakharovProblem2: missing runs 16-20 ---
for run_idx in $(seq 16 20); do
    submit "ExpandedZakharovProblem2" "$run_idx"
done

# --- Problems missing all of runs 6-20 ---
for pname in "GriewankProblem2" "RastriginProblem2" "SalomonProblem2" \
             "SchwefelProblem2" "SphereProblem2" \
             "CrossInTrayProblem" "DropWaveProblem" "EasomProblem" \
             "HimmelblauProblem" "HolderTableProblem" "LeviN13Problem" \
             "MatyasProblem" "SchafferN2Problem" "ThreeHumpCamelProblem"; do
    for run_idx in $(seq 6 20); do
        submit "$pname" "$run_idx"
    done
done

# --- BealeProblem: all 20 runs missing ---
for run_idx in $(seq 1 20); do
    submit "BealeProblem" "$run_idx"
done

# --- GoldsteinPriceProblem: all 20 runs missing ---
for run_idx in $(seq 1 20); do
    submit "GoldsteinPriceProblem" "$run_idx"
done

# --- BoothProblem: bad runs 1,2,5 + missing 6-20 ---
for run_idx in 1 2 5; do
    submit "BoothProblem" "$run_idx"
done
for run_idx in $(seq 6 20); do
    submit "BoothProblem" "$run_idx"
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
