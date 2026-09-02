"""
Round 3 of the metric search (2026-08-03): does the ACTUAL fitted Yeo-Johnson λ
(learned jointly with the rest of the GP hyperparameters during real WarpedGP
BO runs) correlate with the WarpedGP-vs-Standard performance margin — as
opposed to Rounds 1-2's a priori metrics computed from the static proxy
surface alone?

For each problem, loads a handful of real `warpedgp-yja-maxvar_<idx>_problem.jld2`
files from data-warpedgp2/, extracts the fitted λ (first warp param per output
dim — ComposedWarping(YeoJohnsonWarping, AffineWarping) puts YJ first, params
concatenated in composition order, see BOSS.jl output_warping.jl), and computes
max_j |λ_j − 1| per run, then the median across runs.

Read-only with respect to BOSIP experiment data: only reads existing JLD2
files, writes only plots/fitted_warp_lambda.csv (new file, not touching any
existing data or CSV).

Must be run after include("src/main.jl") (needed for JLD2 to reconstruct the
BosipProblem/BossProblem/WarpedGaussianProcess types) and
include("src/compute_acq_scores_final.jl") (for the group problem lists).

Usage (from repo root):
    julia --project=src -e '
        include("src/main.jl");
        include("src/compute_acq_scores_final.jl");
        include("src/extract_fitted_warp_lambda.jl")'
"""

using JLD2
using Statistics: median

const N_RUNS_TO_TRY = 8   # per problem, first N indices that load successfully

function _wgp_dir_for(prob, type)
    type == "BIP" ? joinpath("data-warpedgp2", prob) : joinpath("data-warpedgp2", prob * "_cross")
end

function _fitted_lambda_dev(prob_dir, idx)
    path = joinpath(prob_dir, "warpedgp-yja-maxvar_$(idx)_problem.jld2")
    isfile(path) || return nothing
    try
        f = jldopen(path)
        p = f["problem"]
        close(f)
        warp = p.problem.params.params.warp   # Vector{Vector{Float64}}, one per output dim
        return maximum(abs(w[1] - 1.0) for w in warp)
    catch e
        @warn "Failed to load/parse $path: $e"
        return nothing
    end
end

function _median_fitted_lambda_dev(prob_dir; n=N_RUNS_TO_TRY)
    vals = Float64[]
    for idx in 1:n
        v = _fitted_lambda_dev(prob_dir, idx)
        v !== nothing && push!(vals, v)
    end
    isempty(vals) && return nothing
    return median(vals), length(vals)
end

const _ALL_PROBS_TYPED = vcat(
    [(p, "BIP") for p in _GROUP_A],
    [(p, "Opt") for p in _GROUP_B_PROBS],
    [(p, "BIP") for p in _GROUP_C],
    [(p, "Opt") for p in _GROUP_D_PROBS],
)

function run_extract_fitted_warp_lambda()
    results = NamedTuple[]
    for (prob, typ) in _ALL_PROBS_TYPED
        dir = _wgp_dir_for(prob, typ)
        if !isdir(dir)
            @warn "Missing $dir"
            continue
        end
        r = _median_fitted_lambda_dev(dir)
        if r === nothing
            @warn "No loadable problem files for $prob"
            continue
        end
        med, n = r
        push!(results, (problem=prob, fitted_yj_dev=med, n_runs=n))
        println("$prob: fitted_yj_dev=$(round(med,digits=3)) (n=$n runs)")
    end

    mkpath("plots")
    open("plots/fitted_warp_lambda.csv", "w") do io
        println(io, "problem,fitted_yj_dev,n_runs")
        for r in results
            println(io, "$(r.problem),$(r.fitted_yj_dev),$(r.n_runs)")
        end
    end
    println("\nSaved → plots/fitted_warp_lambda.csv ($(length(results)) problems)")
    return results
end
