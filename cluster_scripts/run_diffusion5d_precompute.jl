# Standalone precompute for DiffusionProblem5D — no CairoMakie.
# CairoMakie (loaded by plot_marginals.jl) hangs in batch jobs; this script
# loads only main.jl + JLD2 + Random, which all work fine in batch.

println("[1] Julia started."); flush(stdout)

include(joinpath(@__DIR__, "..", "src", "main.jl"))
println("[2] main.jl loaded."); flush(stdout)

using JLD2
using Statistics: mean
using Random: randperm
println("[3] Packages loaded."); flush(stdout)

## ── Constants ─────────────────────────────────────────────────────────────────

const _PC_GRID_RES = 50
const _PC_LHC_SIZE = 500
const _PC_GRID_DIR = "data/true_posterior_grids"

_pc_grid_path(problem) = joinpath(_PC_GRID_DIR, get_name(problem) * ".jld2")

## ── LHC sampler ───────────────────────────────────────────────────────────────

function _pc_lhc(lb, ub, n)
    d = length(lb)
    X = Matrix{Float64}(undef, d, n)
    for k in 1:d
        perm    = randperm(n)
        X[k, :] .= lb[k] .+ (ub[k] - lb[k]) .* ((perm .- rand(n)) ./ n)
    end
    return X
end

## ── LHC 1-D marginal ─────────────────────────────────────────────────────────

function _pc_lhc_1d_marginal(logpost, lb, ub, xs_k, dim_k, lhc_size)
    nk  = length(xs_k)
    lhc = _pc_lhc(lb, ub, lhc_size)
    logval = Vector{Float64}(undef, nk)
    Threads.@threads for ik in 1:nk
        col_local = copy(lhc)
        col_local[dim_k, :] .= xs_k[ik]
        lp     = [logpost(col_local[:, j]) for j in 1:lhc_size]
        lp_max = maximum(lp)
        logval[ik] = log(mean(exp.(lp .- lp_max))) + lp_max
    end
    return logval
end

## ── LHC pairwise marginal ─────────────────────────────────────────────────────

function _pc_lhc_marginal(logpost, lb, ub, xs_a, xs_b, dim_a, dim_b, lhc_size)
    na  = length(xs_a)
    nb  = length(xs_b)
    lhc = _pc_lhc(lb, ub, lhc_size)
    logval = Matrix{Float64}(undef, na, nb)
    Threads.@threads for ia in 1:na
        col_local = copy(lhc)
        col_local[dim_a, :] .= xs_a[ia]
        for ib in 1:nb
            col_local[dim_b, :] .= xs_b[ib]
            lp     = [logpost(col_local[:, k]) for k in 1:lhc_size]
            lp_max = maximum(lp)
            logval[ia, ib] = log(mean(exp.(lp .- lp_max))) + lp_max
        end
    end
    return logval
end

## ── Precompute ────────────────────────────────────────────────────────────────

function _pc_precompute(problem; force=false, grid_res=_PC_GRID_RES, lhc_size=_PC_LHC_SIZE)
    name    = get_name(problem)
    outpath = _pc_grid_path(problem)
    d       = x_dim(problem)

    if isfile(outpath) && !force
        println("$name: already exists — skipping (use force=true to recompute)"); flush(stdout)
        return
    end

    res     = grid_res
    lb, ub  = domain(problem).bounds
    logpost = true_logpost(problem)

    xs_per_dim = [collect(range(lb[k], ub[k]; length=res)) for k in 1:d]
    mkpath(_PC_GRID_DIR)

    pairs  = [(da, db) for da in 1:d for db in (da+1):d]
    npairs = length(pairs)

    println("$name: computing $d 1-D marginals (res=$res, lhc=$lhc_size) ..."); flush(stdout)
    marg1d = Vector{Vector{Float64}}(undef, d)
    for k in 1:d
        println("  1D dim $k"); flush(stdout)
        marg1d[k] = _pc_lhc_1d_marginal(logpost, lb, ub, xs_per_dim[k], k, lhc_size)
    end

    println("$name: computing $npairs pairwise marginals (res=$res, lhc=$lhc_size) ..."); flush(stdout)
    marginals = Vector{Matrix{Float64}}(undef, npairs)
    for (pi, (da, db)) in enumerate(pairs)
        println("  pair ($da,$db)"); flush(stdout)
        marginals[pi] = _pc_lhc_marginal(logpost, lb, ub, xs_per_dim[da], xs_per_dim[db], da, db, lhc_size)
    end

    pair_dims = [[da, db] for (da, db) in pairs]
    save(outpath, Dict("dim" => d, "xs_per_dim" => xs_per_dim,
                       "pair_dims" => pair_dims, "pair_marginals" => marginals,
                       "marg1d" => marg1d))

    println("$name: saved to $outpath"); flush(stdout)
end

## ── Entry point ───────────────────────────────────────────────────────────────

const _GRID_RES_ARG  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : _PC_GRID_RES
const _LHC_SIZE_ARG  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : _PC_LHC_SIZE

println("[4] Starting precompute (grid_res=$_GRID_RES_ARG, lhc_size=$_LHC_SIZE_ARG) ..."); flush(stdout)
_pc_precompute(DiffusionProblem5D(); force=true, grid_res=_GRID_RES_ARG, lhc_size=_LHC_SIZE_ARG)
println("Precompute done."); flush(stdout)
