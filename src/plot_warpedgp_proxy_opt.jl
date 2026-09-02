# Compare standard GP vs WarpedGP on BealeProblem and GoldsteinPriceProblem.
# Standard runs (maxvar + eiv, n=5) from data-opt-functions/
# WarpedGP runs (warpedgp-maxvar, n=5) from data-warpedgp/
#
# Output: plots/warpedgp_proxy_opt_tv.{png,pdf}
# Run via include() in a live Julia session from ~/repos/bosip_benchmarks/

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR    = "plots"
const INIT_DATA   = 3
const OPT_DIR     = "data-opt-functions"
const WARPEDGP_DIR = "data-warpedgp"

function load_tv_scores(data_dir, problem, method; max_runs=5)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        fpath = joinpath(data_dir, problem, "$(method)_$(i)_TVmetric.jld2")
        isfile(fpath) || continue
        s = load(fpath, "score")
        isnothing(s) || push!(scores, s)
    end
    return scores
end

function aggregate(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat    = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] = s
    end
    valid = [i for i in 1:maxlen if all(!isnan, mat[i, :])]
    isempty(valid) && return nothing
    xs  = INIT_DATA .+ (valid .- 1)
    med = [median(mat[i, :])            for i in valid]
    lo  = [quantile(mat[i, :], 0.25)   for i in valid]
    hi  = [quantile(mat[i, :], 0.75)   for i in valid]
    return xs, med, lo, hi
end

const METHODS = [
    ("maxvar",          OPT_DIR,      "GP - MaxVar",        Makie.wong_colors()[2]),   # orange
    ("eiv",             OPT_DIR,      "GP - EIV",           Makie.wong_colors()[1]),   # blue
    ("warpedgp-maxvar", WARPEDGP_DIR, "WarpedGP - MaxVar",  Makie.wong_colors()[6]),   # vermillion
]

function add_panel!(figpos, problem_name, title; ylabel=true)
    ax = Axis(figpos;
        title  = title,
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        xscale = log10,
        yscale = log10,
    )
    for (method, data_dir, label, col) in METHODS
        scores = load_tv_scores(data_dir, problem_name, method)
        if isempty(scores)
            @warn "No data: $problem_name / $method"
            continue
        end
        agg = aggregate(scores)
        isnothing(agg) && continue
        xs, med, lo, hi = agg
        band!(ax, xs, lo, hi; color=(col, 0.2))
        lines!(ax, xs, med; color=col, linewidth=2, label=label)
    end
    return ax
end

mkpath(PLOT_DIR)

fig = Figure(; size=(850, 380))

ax1 = add_panel!(fig[1, 1], "BealeProblem",          "Beale";           ylabel=true)
ax2 = add_panel!(fig[1, 2], "GoldsteinPriceProblem", "Goldstein-Price";  ylabel=false)

Legend(fig[1, 3], ax1; tellwidth=true, framevisible=true, labelsize=12,
    title="method (n=5, median ± IQR)")

colgap!(fig.layout, 10)

save(joinpath(PLOT_DIR, "warpedgp_proxy_opt_tv.png"), fig)
save(joinpath(PLOT_DIR, "warpedgp_proxy_opt_tv.pdf"), fig)
@info "Saved to $(PLOT_DIR)/warpedgp_proxy_opt_tv.{png,pdf}"
