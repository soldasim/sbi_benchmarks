#!/bin/bash
# Fix warpedgp runs with < 100 iters (crashes / early stops).
# Audit (2026-07-01, job 11110004) identified 65 bad runs across 10 problems.
# Target: 100 iters (2D rule). Partition: cpu. Memory: 12G (matches original warpedgp budget).

cd ~/repos/bosip_benchmarks

job_ids=()
count=0

submit() {
    local pname="$1" run_idx="$2"
    local job_name="${pname}_wgp_${run_idx}"
    jid=$(sbatch --parsable -p cpu --mem=12G --job-name="$job_name" \
          cluster_scripts/run.sh "$pname" warpedgp-yja-maxvar "$run_idx" 0 100 nothing)
    if [ -n "$jid" ]; then
        job_ids+=("$jid")
        count=$((count + 1))
    else
        echo "FAILED: $job_name"
    fi
}

# RosenbrockProblem2_cross: runs 2(60),5(34),8(32),11(34),12(48),13(79),16(44)
for run_idx in 2 5 8 11 12 13 16; do
    submit "RosenbrockProblem2_cross" "$run_idx"
done

# AckleyProblem2_cross: runs 2(89),3(92),5(61),10(80),11(98),12(83),14(94)
for run_idx in 2 3 5 10 11 12 14; do
    submit "AckleyProblem2_cross" "$run_idx"
done

# ExpandedSchafferF6Problem2_cross: runs 3(77),4(45),6(25),11(75),12(30),13(43),15(42),18(13),19(21),20(86)
for run_idx in 3 4 6 11 12 13 15 18 19 20; do
    submit "ExpandedSchafferF6Problem2_cross" "$run_idx"
done

# SalomonProblem2_cross: runs 15(58),18(75),20(78)
for run_idx in 15 18 20; do
    submit "SalomonProblem2_cross" "$run_idx"
done

# BealeProblem_cross: runs 5(18),6(12),8(64)
for run_idx in 5 6 8; do
    submit "BealeProblem_cross" "$run_idx"
done

# CrossInTrayProblem_cross: runs 1(84),2(56),3(61),4(93),6(77),7(54),8(44),10(93),11(75),12(67),14(84),15(61),16(56),17(42),18(81)
for run_idx in 1 2 3 4 6 7 8 10 11 12 14 15 16 17 18; do
    submit "CrossInTrayProblem_cross" "$run_idx"
done

# DropWaveProblem_cross: run 6(82)
submit "DropWaveProblem_cross" 6

# EasomProblem_cross: runs 2(32),4(25),9(32),10(36),11(23),12(29),13(28),14(77),18(27),20(31)
for run_idx in 2 4 9 10 11 12 13 14 18 20; do
    submit "EasomProblem_cross" "$run_idx"
done

# GoldsteinPriceProblem_cross: run 17(20)
submit "GoldsteinPriceProblem_cross" 17

# LeviN13Problem_cross: runs 2(83),5(42),6(91),10(89),11(39),12(75),13(84),18(85)
for run_idx in 2 5 6 10 11 12 13 18; do
    submit "LeviN13Problem_cross" "$run_idx"
done

echo "Done. $count jobs submitted."
echo "JOB_IDS: ${job_ids[*]}"
