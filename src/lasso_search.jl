"""
LASSO regression across ALL candidate metrics (2026-08-04): checks whether a
SPARSE COMBINATION of predictors explains variance in the margin via automatic
variable selection, rather than testing metrics one at a time as
full_correlation_sweep.jl does. Hand-rolled coordinate-descent LASSO on
standardized predictors (no GLMNet-style dependency in this project's
environment), with 5-fold cross-validation to pick λ.

Run separately for each of the 4 CONTINUOUS targets (WarpedGP fair/AUC,
NonstatGP fair/AUC) — ternary targets are skipped here since LASSO is a
linear-regression tool suited to continuous responses; the ternary sweep is
already covered via Spearman correlation in full_correlation_sweep.jl.

Must be run after include("src/full_correlation_sweep.jl") (for ALL_METRICS /
ALL_METRIC_NAMES) and include("src/correlate_metrics_lib.jl") (for
_read_csv_cols_lib). Read-only w.r.t. existing data; writes only
plots/lasso_search_results.csv.

Usage:
    include("src/correlate_metrics_lib.jl")
    include("src/full_correlation_sweep.jl")
    include("src/lasso_search.jl")
"""

using Statistics: mean, std
using LinearAlgebra: dot
using Random: seed!, shuffle

function _standardize_cols(X)
    mu = vec(mean(X, dims=1))
    sd = vec(std(X, dims=1; corrected=false))
    sd[sd .< 1e-12] .= 1.0
    return (X .- mu') ./ sd', mu, sd
end

_soft(z, γ) = sign(z) * max(abs(z) - γ, 0.0)

function lasso_fit(X, y, λ; max_iter=2000, tol=1e-7)
    n, p = size(X)
    β  = zeros(p)
    yc = y .- mean(y)
    for _ in 1:max_iter
        β_old = copy(β)
        for j in 1:p
            Xj  = @view X[:, j]
            r_j = yc .- X * β .+ Xj .* β[j]
            zj  = dot(Xj, r_j) / n
            denom = dot(Xj, Xj) / n
            β[j] = denom < 1e-12 ? 0.0 : _soft(zj, λ) / denom
        end
        maximum(abs.(β .- β_old)) < tol && break
    end
    return β
end

lasso_predict(X, β, y_mean) = y_mean .+ X * β

function lasso_cv(X, y; nfolds=5, n_lambda=30, seed=42)
    n = size(X, 1)
    λmax = maximum(abs.(X' * (y .- mean(y)))) / n
    λmax <= 0 && return (0.0, [0.0], [0.0])
    λs = exp.(range(log(λmax), log(λmax * 0.01); length=n_lambda))

    seed!(seed)
    perm = shuffle(1:n)
    fold_size = ceil(Int, n / nfolds)
    folds = filter(!isempty, [perm[((k-1)*fold_size+1):min(k*fold_size, n)] for k in 1:nfolds])

    cv_mse = zeros(length(λs))
    for (li, λ) in enumerate(λs)
        errs = Float64[]
        for test_idx in folds
            train_idx = setdiff(1:n, test_idx)
            Xtr, ytr = X[train_idx, :], y[train_idx]
            Xte, yte = X[test_idx, :], y[test_idx]
            β = lasso_fit(Xtr, ytr, λ)
            yhat = lasso_predict(Xte, β, mean(ytr))
            append!(errs, (yte .- yhat) .^ 2)
        end
        cv_mse[li] = mean(errs)
    end
    best_i = argmin(cv_mse)
    return λs[best_i], λs, cv_mse
end

function run_lasso_for_target(target_csv, margin_cols, label)
    m_rows = _read_csv_cols_lib(target_csv, ["problem", margin_cols[1], margin_cols[2]])
    margin_dict = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in m_rows)

    metrics  = [m for m in ALL_METRIC_NAMES if !(startswith(label, "nongp") && m == "fitted_yj_dev")]
    problems = [p for p in keys(margin_dict) if haskey(ALL_METRICS, p) && all(haskey(ALL_METRICS[p], m) for m in metrics)]
    n = length(problems)
    y = [margin_dict[p] for p in problems]
    X = Matrix{Float64}(undef, n, length(metrics))
    for (j, m) in enumerate(metrics)
        X[:, j] = [ALL_METRICS[p][m] for p in problems]
    end

    keep = [length(Set(X[:, j])) > 1 for j in 1:length(metrics)]
    X = X[:, keep]
    metrics = metrics[keep]

    Xs, _, _ = _standardize_cols(X)
    λ_best, _, _ = lasso_cv(Xs, y)
    β = lasso_fit(Xs, y, λ_best)

    nz = [(metrics[j], β[j]) for j in 1:length(metrics) if abs(β[j]) > 1e-8]
    sort!(nz; by = x -> -abs(x[2]))

    println("\n=== LASSO: $label (n=$n, p=$(length(metrics)), λ*=$(round(λ_best,digits=4))) ===")
    if isempty(nz)
        println("  (no predictors survived — LASSO selected the null/intercept-only model)")
    else
        for (m, c) in nz
            println("  $m: β=$(round(c,digits=4))")
        end
    end
    return (label=label, n=n, selected=nz, lambda=λ_best)
end

lasso_results = [
    run_lasso_for_target("plots/warpedgp_scores_fair.csv", ("warpedgp_median","standard_median"), "warpedgp_fair"),
    run_lasso_for_target("plots/warpedgp_scores_auc.csv",  ("warpedgp_median","standard_median"), "warpedgp_auc"),
    run_lasso_for_target("plots/nongp_scores_fair.csv",    ("nongp_median","standard_median"),     "nongp_fair"),
    run_lasso_for_target("plots/nongp_scores_auc.csv",     ("nongp_median","standard_median"),     "nongp_auc"),
]

open("plots/lasso_search_results.csv", "w") do io
    println(io, "target,n,lambda,metric,coefficient")
    for r in lasso_results
        if isempty(r.selected)
            println(io, "$(r.label),$(r.n),$(r.lambda),,")
        else
            for (m, c) in r.selected
                println(io, "$(r.label),$(r.n),$(r.lambda),$(m),$(c)")
            end
        end
    end
end
@info "Saved → plots/lasso_search_results.csv"
