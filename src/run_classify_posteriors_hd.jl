## Runner for the full classify_posteriors pipeline including HD problems (Groups C + D).
## Regenerates plots/classify_posteriors.csv for the 31 original problems,
## then appends 6 HD problems: DuffingProblem5, DiffusionProblem5D,
## Rosenbrock5_cross, StyblinskiTang5_cross, Michalewicz5_cross, Sphere5_cross.
##
## Usage (from interactive Julia session in repos/bosip_benchmarks/):
##   include("src/run_classify_posteriors_hd.jl")

include(joinpath(@__DIR__, "main.jl"))
include(joinpath(@__DIR__, "classify_posteriors.jl"))
include(joinpath(@__DIR__, "classify_posteriors_hd.jl"))
