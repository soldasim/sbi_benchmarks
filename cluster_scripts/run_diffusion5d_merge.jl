# Merge all partial DiffusionProblem5D grid files into the final JLD2.
# Expects all 5 1D files and 10 pair files to exist.

println("[1] Merge started."); flush(stdout)

include(joinpath(@__DIR__, "..", "src", "main.jl"))
println("[2] main.jl loaded."); flush(stdout)

using JLD2
println("[3] JLD2 loaded."); flush(stdout)

const GRID_DIR = "data/true_posterior_grids"
const PAIRS    = [(da, db) for da in 1:5 for db in (da+1):5]
const GRID_RES = 50

prob   = DiffusionProblem5D()
lb, ub = domain(prob).bounds

# Load 1D marginals
println("[4] Loading 1D marginals..."); flush(stdout)
xs_per_dim = [collect(range(lb[k], ub[k]; length=GRID_RES)) for k in 1:5]
marg1d = Vector{Vector{Float64}}(undef, 5)
for k in 1:5
    path = joinpath(GRID_DIR, "DiffusionProblem5D_1d_$(k).jld2")
    isfile(path) || error("Missing 1D file: $path")
    d = load(path)
    marg1d[k] = d["logval"]
    println("  loaded 1D dim $k"); flush(stdout)
end

# Load pairwise marginals
println("[5] Loading pairwise marginals..."); flush(stdout)
pair_dims  = [[da, db] for (da, db) in PAIRS]
marginals  = Vector{Matrix{Float64}}(undef, length(PAIRS))
for (pi, (da, db)) in enumerate(PAIRS)
    path = joinpath(GRID_DIR, "DiffusionProblem5D_pair_$(da)_$(db).jld2")
    isfile(path) || error("Missing pair file: $path")
    d = load(path)
    marginals[pi] = d["logval"]
    println("  loaded pair ($da,$db)"); flush(stdout)
end

# Save final grid
outpath = joinpath(GRID_DIR, "DiffusionProblem5D.jld2")
save(outpath, Dict("dim" => 5, "xs_per_dim" => xs_per_dim,
                   "pair_dims" => pair_dims, "pair_marginals" => marginals,
                   "marg1d" => marg1d))
println("Saved final grid to $outpath"); flush(stdout)
println("Merge done."); flush(stdout)
