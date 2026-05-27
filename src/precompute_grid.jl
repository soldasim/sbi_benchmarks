"""
Precompute grid points for performance metrics.

This script generates and stores grid points used for evaluating performance metrics.
The grids consist of:
- Posterior grid: xs sampled from the prior, plus log-weights and true log-posterior values
- Convergence metrics grid: xs (reused) plus simulator outputs for computing convergence metrics

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

_grid_size(problem::AbstractProblem) = 20 * 10^x_dim(problem)

_grid_size(problem::MultidimProblem) = 20_000
_grid_size(problem::GaussProblem) = 20_000
_grid_size(problem::MeanGauss) = 20_000
_grid_size(problem::RosenbrockProblem) = 20_000
_grid_size(problem::StyblinskiTangProblem) = 20_000
_grid_size(problem::MichalewiczProblem) = 20_000

function precompute_grid(problem::AbstractProblem)
    @info "Precomputing all grids for $(typeof(problem))"
    precompute_posterior_grid(problem)
    precompute_simulator_grid(problem)
    @info "All grids precomputed!"
end

function precompute_posterior_grid(problem::AbstractProblem)
    @info "Precomputing posterior grid for $(typeof(problem))"

    # Create grid directory
    dir = posterior_grid_dir(problem)
    mkpath(dir)

    # Compute grid points
    @info "Generating grid points..."
    xs = rand(x_prior(problem), _grid_size(problem))
    log_ws = 0. .- logpdf.(Ref(x_prior(problem)), eachcol(xs))
    true_logvals = true_logpost(problem).(eachcol(xs))

    # Save to file
    filepath = posterior_grid_filepath(problem)
    @info "Saving grid to $filepath"
    save(filepath, Dict(
        "xs" => xs,
        "log_ws" => log_ws,
        "true_logvals" => true_logvals,
    ))

    @info "Done!"
end

function precompute_simulator_grid(problem::AbstractProblem)
    @info "Precomputing simulator grid for $(typeof(problem))"

    # Create grid directory
    dir = simulator_grid_dir(problem)
    mkpath(dir)

    # Load the xs grid (reuse the existing grid)
    @info "Loading xs grid..."
    grid_filepath_existing = posterior_grid_filepath(problem)
    if !isfile(grid_filepath_existing)
        @warn "Grid file not found at $grid_filepath_existing. Precomputing posterior grid first..."
        precompute_posterior_grid(problem)
    end
    
    grid_data = load(grid_filepath_existing)
    xs = grid_data["xs"]
    log_ws = grid_data["log_ws"]
    
    # Compute simulator outputs at grid points
    @info "Computing simulator outputs at grid points..."
    sim = simulator(problem)
    
    # Evaluate simulator at each sample point (columns are samples)
    sim_outputs = Matrix{Float64}(undef, y_dim(problem), size(xs, 2))
    for i in 1:size(xs, 2)
        sim_outputs[:, i] = sim(xs[:, i])
    end

    # Save to file
    filepath = simulator_grid_filepath(problem)
    @info "Saving simulator grid to $filepath"
    save(filepath, Dict(
        "xs" => xs,
        "log_ws" => log_ws,
        "sim_outputs" => sim_outputs,
    ))

    @info "Done!"
end
