#!/bin/bash
# Resubmit Group B EIV runs with <100 iters that are not currently in queue.
# Target: 100 iters, cpu partition (fits in 24h at ~6.7 min/iter).
# Generated 2026-07-02.

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

submit_one() {
    local pname="$1" run_idx="$2"
    local job_name="${pname}_eiv_${run_idx}"
    echo "Submitting $job_name ..."
    jid=$(sbatch --parsable -p cpu --mem=16G --job-name="$job_name" cluster_scripts/run.sh "$pname" eiv "$run_idx" 0 100 nothing)
    [ -n "$jid" ] && job_ids+=("$jid") && count=$((count + 1)) && echo "  -> $jid" || echo "  -> FAILED"
}

# RosenbrockProblem2_cross: runs 7, 14
submit_one "RosenbrockProblem2_cross" 7
submit_one "RosenbrockProblem2_cross" 14

# StyblinskiTangProblem2_cross: runs 7, 17, 20
submit_one "StyblinskiTangProblem2_cross" 7
submit_one "StyblinskiTangProblem2_cross" 17
submit_one "StyblinskiTangProblem2_cross" 20

# SchwefelProblem2_cross: runs 6, 15, 18
submit_one "SchwefelProblem2_cross" 6
submit_one "SchwefelProblem2_cross" 15
submit_one "SchwefelProblem2_cross" 18

# BoothProblem_cross: runs 5, 12
submit_one "BoothProblem_cross" 5
submit_one "BoothProblem_cross" 12

# HimmelblauProblem_cross: run 20
submit_one "HimmelblauProblem_cross" 20

# HolderTableProblem_cross: runs 9, 20
submit_one "HolderTableProblem_cross" 9
submit_one "HolderTableProblem_cross" 20

# BealeProblem_cross: runs 5, 6, 8, 9, 14, 17
submit_one "BealeProblem_cross" 5
submit_one "BealeProblem_cross" 6
submit_one "BealeProblem_cross" 8
submit_one "BealeProblem_cross" 9
submit_one "BealeProblem_cross" 14
submit_one "BealeProblem_cross" 17

# GoldsteinPriceProblem_cross: runs 9, 14
submit_one "GoldsteinPriceProblem_cross" 9
submit_one "GoldsteinPriceProblem_cross" 14

echo ""
echo "Submitted $count jobs: ${job_ids[*]}"
