"""
Precompute grid points for performance metrics.

This script generates and stores grid points used for evaluating performance metrics.
The grid consists of:
- xs: random samples from the prior
- log_ws: log-weights (negative log-pdf of the prior)
- true_logvals: true log-posterior values

Usage:
    julia precompute_grid.jl <problem>
    
Example:
    julia precompute_grid.jl ABProblem
"""

using BOSS
using BOSIP
using Distributions
using JLD2
using Random

Random.seed!(888) # different seed then in main.jl to avoid identical grid points

include("include_code.jl")

function precompute_grid(problem::AbstractProblem)
    @info "Precomputing grid for $(typeof(problem))"

    # Create grid directory
    dir = grid_dir(problem)
    mkpath(dir)

    # Compute grid points
    @info "Generating grid points..."
    xs = rand(x_prior(problem), _grid_size(problem))
    log_ws = 0. .- logpdf.(Ref(x_prior(problem)), eachcol(xs))
    true_logvals = true_logpost(problem).(eachcol(xs))

    # Save to file
    filepath = grid_filepath(problem)
    @info "Saving grid to $filepath"
    save(filepath, Dict(
        "xs" => xs,
        "log_ws" => log_ws,
        "true_logvals" => true_logvals,
    ))

    @info "Done!"
end

_grid_size(problem::AbstractProblem) = 20 * 10^x_dim(problem)

# _grid_size(problem::MultidimProblem) = 20 * 10^x_dim(problem.problem)
_grid_size(problem::MultidimProblem) = 20_000

_grid_size(problem::GaussProblem) = 20_000
