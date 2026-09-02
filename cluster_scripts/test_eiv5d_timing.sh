#!/bin/bash
# Test run: single EIV run on RosenbrockProblem5 for 5 iterations to measure time per iteration.

cd ~/repos/bosip_benchmarks

sbatch -p cpu --mem=12G --time=02:00:00 --job-name="eiv5d_timing" cluster_scripts/run.sh RosenbrockProblem5 eiv-timing 1 0 5 nothing
