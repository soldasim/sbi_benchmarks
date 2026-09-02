# Compute 1D LHC marginal for one dimension of DiffusionProblem5D.
# ARGS[1] = dim index (1-5)

dim_k = parse(Int, ARGS[1])
println("[1] 1D marginal, dim=$dim_k"); flush(stdout)

include(joinpath(@__DIR__, "..", "src", "main.jl"))
println("[2] main.jl loaded."); flush(stdout)

using JLD2
using Statistics: mean
using Random: randperm
println("[3] Packages loaded."); flush(stdout)

const GRID_RES  = 50
const LHC_SIZE  = 500
const GRID_DIR  = "data/true_posterior_grids"

function _lhc(lb, ub, n)
    d = length(lb)
    X = Matrix{Float64}(undef, d, n)
    for k in 1:d
        perm    = randperm(n)
        X[k, :] .= lb[k] .+ (ub[k] - lb[k]) .* ((perm .- rand(n)) ./ n)
    end
    return X
end

function compute_1d_marginal(logpost, lb, ub, xs_k, dim_k)
    nk  = length(xs_k)
    lhc = _lhc(lb, ub, LHC_SIZE)
    logval = Vector{Float64}(undef, nk)
    Threads.@threads for ik in 1:nk
        col = copy(lhc)
        col[dim_k, :] .= xs_k[ik]
        lp     = [logpost(col[:, j]) for j in 1:LHC_SIZE]
        lp_max = maximum(lp)
        logval[ik] = log(mean(exp.(lp .- lp_max))) + lp_max
    end
    return logval
end

prob    = DiffusionProblem5D()
lb, ub  = domain(prob).bounds
logpost = true_logpost(prob)
xs_k    = collect(range(lb[dim_k], ub[dim_k]; length=GRID_RES))

println("[4] Computing..."); flush(stdout)
logval = compute_1d_marginal(logpost, lb, ub, xs_k, dim_k)

outpath = joinpath(GRID_DIR, "DiffusionProblem5D_1d_$(dim_k).jld2")
mkpath(GRID_DIR)
save(outpath, Dict("dim_k" => dim_k, "xs_k" => xs_k, "logval" => logval))
println("Saved to $outpath"); flush(stdout)
println("Done."); flush(stdout)
