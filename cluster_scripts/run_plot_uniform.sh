#!/bin/bash
export PATH="$HOME/.juliaup/bin:$PATH"
cd ~/repos/bosip_benchmarks
julia --project=src src/plot_uniform.jl
