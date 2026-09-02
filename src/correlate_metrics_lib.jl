"""
Reusable correlation-analysis library for the smoothness-metric search
(2026-08-03 autonomous search for a WarpedGP/NonstatGP-vs-Standard predictor).

`run_correlation(cls_csv, margin_csv, margin_cols, out_prefix; metrics)`
correlates each column in `metrics` (read from `cls_csv`) against
`margin = margin_cols[1] − margin_cols[2]` (read from `margin_csv`), for all
problems present in both files. Prints a table, saves `plots/<out_prefix>.csv`
and a scatter-panel figure `plots/<out_prefix>.{png,pdf}`.

Only reads plots/*.csv classification/score files already produced by other
scripts and writes new plots/<out_prefix>.* files — never touches BOSIP
experiment data (data-bosip-norm/ etc.) or any existing plots/*.csv.
"""

using CairoMakie
using Statistics: mean, std, cor
using Distributions: TDist, ccdf

function _read_csv_cols_lib(path, cols)
    lines  = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

function _rank_lib(v)
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

_spearman_lib(x, y) = cor(_rank_lib(x), _rank_lib(y))

function _pvalue_lib(r, n)
    n <= 2 && return NaN
    abs(r) >= 1.0 && return 0.0
    t = r * sqrt((n - 2) / (1 - r^2))
    return 2 * ccdf(TDist(n - 2), abs(t))
end

# margin_cols = (minuend_col, subtrahend_col); margin = minuend - subtrahend
function run_correlation(cls_csv, margin_csv, margin_cols, out_prefix; metrics)
    cls_rows = _read_csv_cols_lib(cls_csv, ["problem"; metrics])
    cls_dict = Dict(r[1] => Dict(metrics[i] => parse(Float64, r[i+1]) for i in eachindex(metrics)) for r in cls_rows)

    m_rows = _read_csv_cols_lib(margin_csv, ["problem", margin_cols[1], margin_cols[2]])
    margin_dict = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in m_rows)

    problems = [p for p in keys(margin_dict) if haskey(cls_dict, p)]
    @info "$out_prefix: matched $(length(problems)) / $(length(m_rows)) problems"
    if isempty(problems)
        @warn "$out_prefix: no matched problems, skipping"
        return NamedTuple[]
    end

    margins = [margin_dict[p] for p in problems]

    println("\n=== $out_prefix (n=$(length(problems))) ===")
    println(rpad("Metric", 12), "│  Pearson r │   p-value │  Spearman ρ │   p-value")
    corr_results = NamedTuple[]
    for m in metrics
        vals = [cls_dict[p][m] for p in problems]
        r  = cor(vals, margins)
        rp = _pvalue_lib(r, length(vals))
        ρ  = _spearman_lib(vals, margins)
        ρp = _pvalue_lib(ρ, length(vals))
        push!(corr_results, (metric=m, pearson=r, pearson_p=rp, spearman=ρ, spearman_p=ρp, n=length(vals)))
        flag_r = rp < 0.05 ? "*" : " "
        flag_ρ = ρp < 0.05 ? "*" : " "
        println(rpad(m, 12), "│  ", lpad(round(r, digits=3), 8), flag_r, " │  ",
                lpad(round(rp, digits=3), 7), " │  ", lpad(round(ρ, digits=3), 9), flag_ρ,
                " │  ", lpad(round(ρp, digits=3), 7))
    end

    mkpath("plots")
    open("plots/$(out_prefix).csv", "w") do io
        println(io, "metric,pearson_r,pearson_p,spearman_rho,spearman_p,n")
        for r in corr_results
            println(io, "$(r.metric),$(r.pearson),$(r.pearson_p),$(r.spearman),$(r.spearman_p),$(r.n)")
        end
    end

    fig = Figure(; size = (340*length(metrics), 380))
    for (i, m) in enumerate(metrics)
        vals = [cls_dict[p][m] for p in problems]
        ax = Axis(fig[1, i]; xlabel = m,
            ylabel = i == 1 ? "margin" : "",
            title  = "r=$(round(corr_results[i].pearson, digits=2))  ρ=$(round(corr_results[i].spearman, digits=2))",
            titlesize = 11)
        scatter!(ax, vals, margins; color = (:steelblue, 0.7), markersize = 8)
        hlines!(ax, [0.0]; color = (:black, 0.3), linestyle = :dash, linewidth = 1)
        xb, yb = mean(vals), mean(margins)
        denom = sum((vals .- xb) .^ 2)
        if denom > 1e-12
            b = sum((vals .- xb) .* (margins .- yb)) / denom
            a = yb - b * xb
            xs = range(minimum(vals), maximum(vals); length=2)
            lines!(ax, xs, a .+ b .* xs; color = :firebrick, linewidth = 2)
        end
    end
    save("plots/$(out_prefix).png", fig; px_per_unit = 3)
    save("plots/$(out_prefix).pdf", fig)
    @info "Saved plots/$(out_prefix).{csv,png,pdf}"

    return corr_results
end

# Multivariate OLS R² (and adjusted R²) of margin ~ metrics, for checking
# whether a COMBINATION of metrics explains variance even if no single one
# correlates individually. High-predictor-count / small-n overfitting risk is
# real here — adjusted R² partially accounts for it but this is a rough check,
# not a rigorous model.
function multivariate_r2(margin_csv, margin_cols, metrics_per_csv)
    # metrics_per_csv :: Vector{(csv_path, [colnames...], name_prefix)}
    all_dicts = Dict{String, Dict{String,Float64}}()
    colnames  = String[]
    for (csv, cols, prefix) in metrics_per_csv
        rows = _read_csv_cols_lib(csv, ["problem"; cols])
        for r in rows
            d = get!(all_dicts, r[1], Dict{String,Float64}())
            for i in eachindex(cols)
                d["$(prefix)$(cols[i])"] = parse(Float64, r[i+1])
            end
        end
        append!(colnames, ["$(prefix)$(c)" for c in cols])
    end

    m_rows = _read_csv_cols_lib(margin_csv, ["problem", margin_cols[1], margin_cols[2]])
    margin_dict = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in m_rows)

    problems = [p for p in keys(margin_dict) if haskey(all_dicts, p) && all(haskey(all_dicts[p], c) for c in colnames)]
    n = length(problems)
    k = length(colnames)
    @info "multivariate_r2: n=$n problems, k=$k predictors ($(join(colnames, ", ")))"
    n <= k + 1 && (@warn "multivariate_r2: n <= k+1, regression is degenerate/perfect fit, skipping"; return nothing)

    y = [margin_dict[p] for p in problems]
    Xd = Matrix{Float64}(undef, n, k + 1)
    Xd[:, 1] .= 1.0
    for (j, c) in enumerate(colnames)
        Xd[:, j+1] = [all_dicts[p][c] for p in problems]
    end

    beta = Xd \ y
    yhat = Xd * beta
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    println("\n=== Multivariate OLS: margin ~ [$(join(colnames, ", "))] ===")
    println("n=$n, k=$k predictors, R²=$(round(r2, digits=3)), adjusted R²=$(round(adj_r2, digits=3))")
    return (n=n, k=k, r2=r2, adj_r2=adj_r2, colnames=colnames, beta=beta)
end
