# Generate random starts and precompute metric grids for the IMMD benchmark problems.
# Data is stored in data-bosip-norm/.
# Run on a compute node (not the login node).
#
# ProxySIR grid is pre-copied from data-bosip — only starts are generated for ProxySIRProblem.
# All other problems have both grids and starts generated here.

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

analytical_problems = [
    ABProblem(),
    SimpleProblem(),
    BananaProblem(),
    BimodalProblem(),
]

pde_problems = [
    DuffingProblem(),
    DiffusionProblem10(),
]

proxysir = ProxySIRProblem()

# Analytical problems: grids + starts
for problem in analytical_problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

# ProxySIR: grid already exists in data-bosip — copy it, then generate starts
@info "Setting up ProxySIRProblem (copying grid from data-bosip, generating starts) ..."
proxysir_grid_src = "data-bosip/" * get_name(proxysir) * "/grid"
proxysir_grid_dst = data_dir(proxysir) * "/grid"
mkpath(proxysir_grid_dst)
for f in readdir(proxysir_grid_src)
    cp(joinpath(proxysir_grid_src, f), joinpath(proxysir_grid_dst, f); force=true)
end
generate_starts(proxysir, 20)
@info "Done: ProxySIRProblem"

# PDE problems: grids + starts (slow — Diffusion may take several hours)
for problem in pde_problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "All 7 IMMD benchmark problems set up."
