#!/bin/bash
# TEST JOB: Submit a single warpedgp-yja-maxvar run for DuffingProblem5 (Group C).
# Purpose: verify the warpedgp script works on 5D BIP problems before bulk submission.
# Data stored in data-warpedgp2/DuffingProblem5/.

cd ~/repos/bosip_benchmarks

echo "Submitting test job: DuffingProblem5 warpedgp-yja-maxvar run 1 ..."
jid=$(sbatch --parsable -p cpulong --mem=12G --job-name="DuffingProblem5_warpedgp-yja_1" \
    cluster_scripts/run.sh DuffingProblem5 warpedgp-yja-maxvar 1 0 100 nothing)
echo "Submitted job: $jid"
echo "Monitor with: tail -f slurm-${jid}.out"
