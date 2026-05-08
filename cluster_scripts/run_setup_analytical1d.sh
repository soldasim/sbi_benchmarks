#!/bin/bash
export PATH="$HOME/.juliaup/bin:$PATH"
cd ~/repos/bosip_benchmarks
julia --project=src src/setup_analytical1d.jl
