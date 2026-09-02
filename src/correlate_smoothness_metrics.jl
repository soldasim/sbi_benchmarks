"""
Correlate each of the 5 smoothness-classification metrics (cv_value, cv_grad,
yj_dev, skewness, heterosced — see classify_smoothness.jl) against the actual
WarpedGP-vs-Standard performance margin (warpedgp_scores_fair.csv), to see
which metric (if any) predicts when WarpedGP actually helps.

Margin = warpedgp_median − standard_median (median final log-TV). Negative =
WarpedGP wins. If a metric is a good predictor, higher metric values should
correlate with more negative margins (i.e. a NEGATIVE correlation coefficient).

Reports both Pearson r (linear) and Spearman ρ (monotonic, robust to outliers/
grid-boundary saturation in yj_dev) with a two-sided p-value (t-approximation),
and saves a 5-panel scatter (metric vs. margin, with an OLS fit line) to
plots/smoothness_metric_correlations.{png,pdf}.

Data is read dynamically from:
  plots/classify_smoothness.csv   — the 5 metrics per problem
  plots/warpedgp_scores_fair.csv  — standard_median, warpedgp_median (37 problems)

Usage (from repo root):
    julia --project=src -e 'include("src/correlate_smoothness_metrics.jl")'
"""

using CairoMakie
using Statistics: mean, std, cor
using Distributions: TDist, ccdf

## ── Load data ────────────────────────────────────────────────────────────────

function _read_csv_cols_corr(path, cols)
    lines  = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

const METRICS = ["cv_value", "cv_grad", "yj_dev", "skewness", "heterosced"]

_cls_rows = _read_csv_cols_corr("plots/classify_smoothness.csv", ["problem"; METRICS])
_cls_dict = Dict(r[1] => Dict(METRICS[i] => parse(Float64, r[i+1]) for i in eachindex(METRICS)) for r in _cls_rows)

_wgp_rows = _read_csv_cols_corr("plots/warpedgp_scores_fair.csv",
                                 ["problem", "warpedgp_median", "standard_median"])
_margin_dict = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in _wgp_rows)

const PROBLEMS = [p for p in keys(_margin_dict) if haskey(_cls_dict, p)]
@info "Matched $(length(PROBLEMS)) / $(length(_wgp_rows)) problems between the two CSVs"

## ── Rank-based (Spearman) correlation ─────────────────────────────────────────

function _rank(v)
    n   = length(v)
    idx = sortperm(v)
    ranks = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[idx[j+1]] == v[idx[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            ranks[idx[k]] = avg
        end
        i = j + 1
    end
    return ranks
end

_spearman(x, y) = cor(_rank(x), _rank(y))

function _pvalue(r, n)
    n <= 2 && return NaN
    (abs(r) >= 1.0) && return 0.0
    t = r * sqrt((n - 2) / (1 - r^2))
    return 2 * ccdf(TDist(n - 2), abs(t))
end

## ── Compute correlations ──────────────────────────────────────────────────────

margins = [_margin_dict[p] for p in PROBLEMS]

println()
println(rpad("Metric", 12), "│  Pearson r │   p-value │  Spearman ρ │   p-value")
println(repeat('─', 12), "─┼────────────┼───────────┼─────────────┼──────────")
corr_results = NamedTuple[]
for m in METRICS
    vals = [_cls_dict[p][m] for p in PROBLEMS]
    r  = cor(vals, margins)
    rp = _pvalue(r, length(vals))
    ρ  = _spearman(vals, margins)
    ρp = _pvalue(ρ, length(vals))
    push!(corr_results, (metric=m, pearson=r, pearson_p=rp, spearman=ρ, spearman_p=ρp))
    flag_r = rp < 0.05 ? "*" : " "
    flag_ρ = ρp < 0.05 ? "*" : " "
    println(rpad(m, 12), "│  ", lpad(round(r, digits=3), 8), flag_r, " │  ",
            lpad(round(rp, digits=3), 7), " │  ", lpad(round(ρ, digits=3), 9), flag_ρ,
            " │  ", lpad(round(ρp, digits=3), 7))
end
println("\n(Negative correlation = higher metric value predicts WarpedGP winning, as hypothesized. * = p<0.05, n=$(length(PROBLEMS)))")

mkpath("plots")
open("plots/smoothness_metric_correlations.csv", "w") do io
    println(io, "metric,pearson_r,pearson_p,spearman_rho,spearman_p")
    for r in corr_results
        println(io, "$(r.metric),$(r.pearson),$(r.pearson_p),$(r.spearman),$(r.spearman_p)")
    end
end
@info "Saved → plots/smoothness_metric_correlations.csv"

## ── Scatter panel per metric ──────────────────────────────────────────────────

fig_corr = Figure(; size = (1700, 380))

for (i, m) in enumerate(METRICS)
    vals = [_cls_dict[p][m] for p in PROBLEMS]
    ax = Axis(fig_corr[1, i];
        xlabel = m,
        ylabel = i == 1 ? "WarpedGP margin\n(neg = WarpedGP wins)" : "",
        title  = "r=$(round(corr_results[i].pearson, digits=2))  ρ=$(round(corr_results[i].spearman, digits=2))",
        titlesize = 12,
    )
    scatter!(ax, vals, margins; color = (:steelblue, 0.7), markersize = 8)
    hlines!(ax, [0.0]; color = (:black, 0.3), linestyle = :dash, linewidth = 1)

    # OLS fit line
    xb, yb = mean(vals), mean(margins)
    b = sum((vals .- xb) .* (margins .- yb)) / sum((vals .- xb) .^ 2)
    a = yb - b * xb
    xs = range(minimum(vals), maximum(vals); length=2)
    lines!(ax, xs, a .+ b .* xs; color = :firebrick, linewidth = 2)
end

save("plots/smoothness_metric_correlations.png", fig_corr; px_per_unit = 3)
save("plots/smoothness_metric_correlations.pdf", fig_corr)
@info "Saved → plots/smoothness_metric_correlations.{png,pdf}"
