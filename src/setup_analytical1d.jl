# Generate starts and precompute grids for all new 1D analytical problems.
# Run on a compute node (not login node).

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

problems = vcat(
    [MultidimProblem(SquareProblem(), d) for d in 1:6],
    [MultidimProblem(SineProblem(),   d) for d in 1:6],
    [MultidimProblem(CubicProblem(),  d) for d in 1:6],
)

for problem in problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)        # 20 starts (same as other problems)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "All problems set up."
