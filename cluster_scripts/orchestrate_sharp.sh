#!/bin/bash
# Orchestrates the full sharp-experiments pipeline:
#   1. Setup (starts + grids)
#   2. MaxVar + EIV experiments (240 jobs, dependent on setup)
#   3. Notify when experiments done
#   4. TV-metric plot + progression plots (dependent on experiments)
#   5. Notify when plots done
#
# Usage: bash cluster_scripts/orchestrate_sharp.sh
# Run from ~/repos/bosip_benchmarks

set -e
cd ~/repos/bosip_benchmarks

# ── 1. Setup ──────────────────────────────────────────────────────────────────
echo "Submitting setup job..."
SETUP_JID=$(sbatch --parsable cluster_scripts/setup_sharp_opt_problems.sh)
echo "  setup job: $SETUP_JID"

# ── 2. Experiments ────────────────────────────────────────────────────────────
echo "Submitting MaxVar and EIV experiments (depend on setup)..."

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

exp_job_ids=()
for pname in "${problems[@]}"; do
    for run_idx in 1 2 3 4 5; do
        for acq in maxvar eiv; do
            jid=$(sbatch --parsable --dependency=afterok:$SETUP_JID \
                -p cpu --mem=12G \
                --job-name="${pname}_${acq}_${run_idx}" \
                cluster_scripts/run.sh "$pname" "$acq" "$run_idx" 0 200 nothing)
            exp_job_ids+=("$jid")
        done
    done
done

echo "  submitted ${#exp_job_ids[@]} experiment jobs"

# ── 3. Notify when experiments done ───────────────────────────────────────────
dep=$(IFS=:; echo "${exp_job_ids[*]}")
NOTIFY1_JID=$(sbatch --parsable --dependency=afterany:$dep \
    -p cpu --mem=1G --job-name=notify_sharp_exp \
    --wrap="touch ~/repos/bosip_benchmarks/.sharp_experiments_done")
echo "  notify-1 job: $NOTIFY1_JID"

# ── 4. Plots ──────────────────────────────────────────────────────────────────
PLOT_RESULTS_JID=$(sbatch --parsable --dependency=afterany:$dep \
    cluster_scripts/run_plot_sharp_results.sh)
PLOT_PROG_JID=$(sbatch --parsable --dependency=afterany:$dep \
    cluster_scripts/run_plot_sharp_progressions.sh)
echo "  plot-results job: $PLOT_RESULTS_JID"
echo "  plot-progressions job: $PLOT_PROG_JID"

# ── 5. Notify when plots done ─────────────────────────────────────────────────
NOTIFY2_JID=$(sbatch --parsable \
    --dependency=afterany:${PLOT_RESULTS_JID}:${PLOT_PROG_JID} \
    -p cpu --mem=1G --job-name=notify_sharp_plots \
    --wrap="touch ~/repos/bosip_benchmarks/.sharp_plots_done")
echo "  notify-2 job: $NOTIFY2_JID"

echo ""
echo "Pipeline submitted. Setup → Experiments → Plots."
echo "Monitor with: squeue -u \$USER"
