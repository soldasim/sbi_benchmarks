"""
AUC-based margin variant (2026-08-04): instead of scoring each run at a single
point (its last valid TV reading within the shared iteration budget, as
compute_fair_scores.jl does), this integrates over the WHOLE shared-budget
trajectory: AUC = mean(log(TV)) over all valid entries in score[1:T]. More
robust to a single noisy endpoint reading than the last-point score.

Mirrors compute_fair_scores.jl's per-problem group/directory routing and
shared-iteration-budget (T = min raw TV-history length across both groups)
exactly — only the per-run scoring function differs (mean over the trajectory
instead of last-valid-point).

## Outputs
  plots/warpedgp_scores_auc.csv  (37 problems; columns include shared_T)
  plots/nongp_scores_auc.csv     (as many problems as compute_nongp_scores_fair
                                    covers; DiffusionProblem5D excludes the same
                                    6 crashed nongp runs, see compute_fair_scores.jl)

## Usage
    include("src/compute_acq_scores_final.jl")   # for group lists
    include("src/compute_fair_scores.jl")        # for _collect_score_vecs, _DIFFUSION5D_NONGP_CRASHED, _collect_score_vecs_excl
    include("src/compute_auc_margin.jl")
"""

using Statistics: mean, median

# Mean log-TV over the valid (finite, >0) entries within score[1:T].
function _auc_score_at_budget(score, T)
    Tc = min(T, length(score))
    valid = Float64[score[k] for k in 1:Tc if isfinite(score[k]) && score[k] > 0.0]
    isempty(valid) && return nothing
    return mean(log.(valid))
end

function _auc_pair_scores(std_vecs, other_vecs)
    if isempty(std_vecs) || isempty(other_vecs)
        return (0, Float64[], Float64[])
    end
    T = minimum(length(v) for v in Iterators.flatten((std_vecs, other_vecs)))
    std_scores = Float64[]
    for v in std_vecs
        s = _auc_score_at_budget(v, T)
        s === nothing || push!(std_scores, s)
    end
    other_scores = Float64[]
    for v in other_vecs
        s = _auc_score_at_budget(v, T)
        s === nothing || push!(other_scores, s)
    end
    return (T, std_scores, other_scores)
end

function compute_warpedgp_scores_auc()
    rows = FairRow[]

    function add_row!(prob, typ, std_dir, std_key, std_n, wgp_dir)
        isdir(std_dir) || (@warn "Missing: $std_dir"; return)
        isdir(wgp_dir) || (@warn "Missing: $wgp_dir"; return)
        std_vecs = _collect_score_vecs(std_dir, std_key;              n=std_n)
        wgp_vecs = _collect_score_vecs(wgp_dir, "warpedgp-yja-maxvar"; n=20)
        T, s_scores, w_scores = _auc_pair_scores(std_vecs, wgp_vecs)
        T == 0 && (@warn "No data for $prob"; return)
        s_med = isempty(s_scores) ? NaN : median(s_scores)
        w_med = isempty(w_scores) ? NaN : median(w_scores)
        w = _winner([("Standard",s_med),("WarpedGP",w_med)])
        push!(rows, FairRow(prob, typ, T, length(s_scores),s_med, length(w_scores),w_med, w))
        @info "WGP-auc $prob  T=$T std=$(length(s_scores)) wgp=$(length(w_scores)) → $w"
    end

    for prob in _GROUP_A
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-warpedgp2", prob))
    end
    for prob in _GROUP_B_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-warpedgp2", prob * "_cross"))
    end
    for prob in _GROUP_C
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-warpedgp2", prob))
    end
    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-warpedgp2", prob * "_cross"))
    end

    mkpath("plots")
    open("plots/warpedgp_scores_auc.csv", "w") do io
        println(io, "problem,type,shared_T,standard_n,standard_median,warpedgp_n,warpedgp_median,winner")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.shared_T),$(r.standard_n),$(r.standard_med),$(r.other_n),$(r.other_med),$(r.winner)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/warpedgp_scores_auc.csv"
    return rows
end

function compute_nongp_scores_auc()
    rows = FairRow[]

    function add_row!(prob, typ, std_dir, std_key, std_n, nongp_dir; exclude_nongp::Vector{Int}=Int[])
        isdir(std_dir)   || (@warn "Missing: $std_dir"; return)
        isdir(nongp_dir) || (@warn "Missing: $nongp_dir"; return)
        std_vecs   = _collect_score_vecs(std_dir, std_key; n=std_n)
        nongp_vecs = _collect_score_vecs_excl(nongp_dir, "nongp"; n=20, exclude=exclude_nongp)
        T, s_scores, n_scores = _auc_pair_scores(std_vecs, nongp_vecs)
        T == 0 && (@warn "No data for $prob"; return)
        s_med = isempty(s_scores) ? NaN : median(s_scores)
        n_med = isempty(n_scores) ? NaN : median(n_scores)
        w = _winner([("Standard",s_med),("NonstatGP",n_med)])
        push!(rows, FairRow(prob, typ, T, length(s_scores),s_med, length(n_scores),n_med, w))
        @info "NonGP-auc $prob  T=$T std=$(length(s_scores)) nongp=$(length(n_scores)) → $w"
    end

    for prob in _GROUP_A
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-bosip-norm", prob))
    end
    for prob in _GROUP_B_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-opt-functions", prob * "_cross"))
    end
    for prob in _GROUP_C
        excl = prob == "DiffusionProblem5D" ? _DIFFUSION5D_NONGP_CRASHED : Int[]
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-bosip-norm", prob); exclude_nongp=excl)
    end
    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-opt-functions", prob * "_cross"))
    end

    mkpath("plots")
    open("plots/nongp_scores_auc.csv", "w") do io
        println(io, "problem,type,shared_T,standard_n,standard_median,nongp_n,nongp_median,winner")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.shared_T),$(r.standard_n),$(r.standard_med),$(r.other_n),$(r.other_med),$(r.winner)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/nongp_scores_auc.csv"
    return rows
end
