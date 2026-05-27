#!/bin/bash
export PATH="$HOME/.juliaup/bin:$PATH"
cd ~/repos/bosip_benchmarks
julia --project=src -e 'using Pkg; Pkg.precompile()'
