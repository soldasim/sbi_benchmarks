#!/bin/bash
cd /home/soldasim/repos/bosip_benchmarks
export PATH="$HOME/.juliaup/bin:$PATH"
srun --pty --time=02:00:00 --mem=16G --cpus-per-task=4 -p cpufast bash -c \
  'cd /home/soldasim/repos/bosip_benchmarks && export PATH=$HOME/.juliaup/bin:$PATH && julia --project=src src/setup_hex_all_problems.jl && echo done > hex_all_setup.done'
