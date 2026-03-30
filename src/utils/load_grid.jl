"""
Utility functions for loading precomputed grid data.
"""

"""
    load_grid(problem::AbstractProblem)

Load precomputed grid points for the given problem.

Returns a named tuple with fields:
- `xs`: Grid points sampled from the prior
- `log_ws`: Log-weights (negative log prior densities)
- `true_logvals`: True log-posterior values

Throws an error if the grid file doesn't exist.
"""
function load_grid(problem::AbstractProblem)
    filepath = grid_filepath(problem)
    
    if !isfile(filepath)
        error("""
        Grid file not found: $filepath
        
        Please run the precompute_grid.jl script first:
            julia precompute_grid.jl $(typeof(problem) |> string)
        """)
    end
    
    data = load(filepath)
    return (
        xs = data["xs"],
        log_ws = data["log_ws"],
        true_logvals = data["true_logvals"],
    )
end
