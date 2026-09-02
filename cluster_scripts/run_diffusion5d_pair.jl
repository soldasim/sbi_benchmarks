# Compute pairwise LHC marginal for one (da,db) pair of DiffusionProblem5D.
# ARGS[1] = pair index 1-10 (maps to (da,db) pairs for d=5)

const PAIRS = [(da, db) for da in 1:5 for db in (da+1):5]
pair_idx    = parse(Int, ARGS[1])
da, db      = PAIRS[pair_idx]
println("[1] Pair ($da,$db) [index $pair_idx]"); flush(stdout)

include(joinpath(@__DIR__, "..", "src", "main.jl"))
println("[2] main.jl loaded."); flush(stdout)

using JLD2
using Statistics: mean
using Random: randperm
println("[3] Packages loaded."); flush(stdout)

const GRID_RES = 50
const LHC_SIZE = 500
const GRID_DIR = "data/true_posterior_grids"

function _lhc(lb, ub, n)
    d = length(lb)
    X = Matrix{Float64}(undef, d, n)
    for k in 1:d
        perm    = randperm(n)
        X[k, :] .= lb[k] .+ (ub[k] - lb[k]) .* ((perm .- rand(n)) ./ n)
    end
    return X
end

function compute_pair_marginal(logpost, lb, ub, xs_a, xs_b, dim_a, dim_b)
    na  = length(xs_a)
    nb  = length(xs_b)
    lhc = _lhc(lb, ub, LHC_SIZE)
    logval = Matrix{Float64}(undef, na, nb)
    Threads.@threads for ia in 1:na
        col = copy(lhc)
        col[dim_a, :] .= xs_a[ia]
        for ib in 1:nb
            col[dim_b, :] .= xs_b[ib]
            lp     = [logpost(col[:, k]) for k in 1:LHC_SIZE]
            lp_max = maximum(lp)
            logval[ia, ib] = log(mean(exp.(lp .- lp_max))) + lp_max
        end
    end
    return logval
end

prob    = DiffusionProblem5D()
lb, ub  = domain(prob).bounds
logpost = true_logpost(prob)
xs_a    = collect(range(lb[da], ub[da]; length=GRID_RES))
xs_b    = collect(range(lb[db], ub[db]; length=GRID_RES))

println("[4] Computing pair ($da,$db)..."); flush(stdout)
logval = compute_pair_marginal(logpost, lb, ub, xs_a, xs_b, da, db)

outpath = joinpath(GRID_DIR, "DiffusionProblem5D_pair_$(da)_$(db).jld2")
mkpath(GRID_DIR)
save(outpath, Dict("da" => da, "db" => db, "xs_a" => xs_a, "xs_b" => xs_b, "logval" => logval))
println("Saved to $outpath"); flush(stdout)
println("Done."); flush(stdout)
