## Runner for classify_smoothness.jl — computes cv_value and cv_grad for all
## 37 problems (31 original + 6 HD) and writes plots/classify_smoothness.csv.
##
## Usage (from interactive Julia session in repos/bosip_benchmarks/):
##   include("src/run_classify_smoothness.jl")

include(joinpath(@__DIR__, "main.jl"))
include(joinpath(@__DIR__, "classify_smoothness.jl"))

run_classify_smoothness()
