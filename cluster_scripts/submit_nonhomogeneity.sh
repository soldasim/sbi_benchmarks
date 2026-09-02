#!/bin/bash
# Submit the non-homogeneity criterion H (paper1_suggestions.md Suggestion 1) for all
# 37 benchmark problems, split per (problem, output dimension) — 177 jobs total.
# Worst-case single job (a dx=5 problem's dimension, multistart=24 everywhere,
# N=200/m=20/B=200) measured/estimated at ~7.4h; most jobs (dx=2) are ~9 min.
# `cpu` partition (1 day) comfortably covers the worst case.
#
# Reads cluster_scripts/nonhomogeneity_problems.txt ("<reconstructible_name> <y_dim>"
# per line, generated from _SM_ALL_PROBLEMS — regenerate it if the problem set changes).
#
# Run from ~/repos/bosip_benchmarks.

cd ~/repos/bosip_benchmarks

queued=$(squeue -u soldasim --noheader -o "%j" 2>/dev/null)

job_ids=()
count=0
skipped=0
while read -r pname ydim; do
    [ -z "$pname" ] && continue
    for dim in $(seq 1 "$ydim"); do
        job_name="nonhomog_${pname}_dim${dim}"
        out_file="plots/nonhomogeneity/${pname}_dim${dim}.jld2"
        if [ -f "$out_file" ]; then
            echo "Skipping $job_name (output already exists: $out_file)"
            skipped=$((skipped + 1))
            continue
        fi
        if echo "$queued" | grep -qx "$job_name"; then
            echo "Skipping $job_name (already queued/running)"
            skipped=$((skipped + 1))
            continue
        fi
        # One-off exclusion: covered by an already-running debug-test job (2026-08-31,
        # job 11449876, name suffixed _TEST so it doesn't match the exact-name check
        # above) validating the dx=5 path before this bulk submission. Remove this
        # block once that job completes and is no longer needed as a special case.
        if [ "$pname" = "RosenbrockProblem5_cross" ] && [ "$dim" = "1" ]; then
            echo "Skipping $job_name (covered by in-flight debug-test job 11449876)"
            skipped=$((skipped + 1))
            continue
        fi
        echo "Submitting $job_name ..."
        jid=$(sbatch --parsable -p cpu --mem=8G --job-name="$job_name" \
            cluster_scripts/run_nonhomogeneity.sh "$pname" "$dim")
        job_ids+=("$jid")
        count=$((count + 1))
    done
done < cluster_scripts/nonhomogeneity_problems.txt

echo "Done. $count jobs submitted, $skipped skipped (already queued)."
echo "JOB_IDS: ${job_ids[*]}"
