# 3×2 comparison: without proxy (left) vs with proxy (right)
# Rows: SIR, Beale, Goldstein-Price
#
# SIR data lives in data-bosip/ (method names: standard, eiv)
# Beale / Goldstein data live in data-opt-functions/ (method names: maxvar, eiv)
#
# Run from: ~/repos/bosip_benchmarks/
#   julia --project=src src/plot_proxy_comparison.jl

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR = "plots"
const INIT_DATA = 3

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

function load_tv_scores(data_dir::String, problem::String, method::String, run_idx::Int)
    fpath = joinpath(data_dir, problem, "$(method)_$(run_idx)_TVmetric.jld2")
    isfile(fpath) || return nothing
    return load(fpath, "score")
end

function load_all_tv_scores(data_dir::String, problem::String, method::String; max_runs::Int=20)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        s = load_tv_scores(data_dir, problem, method, i)
        isnothing(s) || push!(scores, s)
    end
    return scores
end

function median_scores(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] = s
    end
    xs  = (INIT_DATA):(INIT_DATA + maxlen - 1)
    med = [median(filter(!isnan, mat[i, :])) for i in 1:maxlen]
    return collect(xs), med
end

# ─────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────

const PALETTE = Dict(
    "maxvar"   => Makie.wong_colors()[2],   # orange
    "standard" => Makie.wong_colors()[2],   # orange (same role as MaxVar)
    "eiv"      => Makie.wong_colors()[1],   # blue
)

method_label(m) = Dict(
    "standard" => "MaxVar",
    "maxvar"   => "MaxVar",
    "eiv"      => "EIV",
)[m]

# ─────────────────────────────────────────────
# Panel plotting
# ─────────────────────────────────────────────

function add_tv_panel!(figpos, data_dir::String, problem::String, methods::Vector{String};
                       title="", legend=false, ylabel=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV" : "",
        title  = title,
        xscale = log10,
        yscale = log10,
    )

    for method in methods
        scores = load_all_tv_scores(data_dir, problem, method)
        if isempty(scores)
            @warn "No data for $problem / $method"
            continue
        end
        col = get(PALETTE, method, :gray)
        lbl = method_label(method)
        n   = length(scores)
        maxlen = maximum(length.(scores))
        xs = collect((INIT_DATA):(INIT_DATA + maxlen - 1))

        for s in scores
            lines!(ax, xs[eachindex(s)], s; color=col, alpha=0.8, linewidth=0.5)
        end

        agg = median_scores(scores)
        isnothing(agg) && continue
        xs_med, med = agg
        lines!(ax, xs_med, med; color=col, linewidth=2, label="$lbl (n=$n)")
    end

    legend && axislegend(ax; position=:lb, labelsize=11)
    return ax
end

# ─────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────

# Cell definitions: (data_dir, problem_name, methods, title)
cells = [
    # Row 1: SIR
    ("data-bosip",        "SIRProblem",                 ["standard", "eiv"], "SIR (no proxy)"),
    ("data-bosip",        "ProxySIRProblem",            ["standard"],        "SIR (proxy)"),
    # Row 2: Beale
    ("data-opt-functions", "BealeProblem",               ["maxvar", "eiv"],   "Beale (no proxy)"),
    ("data-opt-functions", "BealeProxyProblem",          ["maxvar", "eiv"],   "Beale (proxy)"),
    # Row 3: Goldstein-Price
    ("data-opt-functions", "GoldsteinPriceProblem",      ["maxvar", "eiv"],   "Goldstein-Price (no proxy)"),
    ("data-opt-functions", "GoldsteinPriceProxyProblem", ["maxvar", "eiv"],   "Goldstein-Price (proxy)"),
]

positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]

ax_w, ax_h = 420, 320
fig = Figure(; size = (ax_w * 2 + 60, ax_h * 3 + 60))

Label(fig[0, 1]; text="Without proxy", font=:bold, fontsize=15, tellwidth=false)
Label(fig[0, 2]; text="With proxy",    font=:bold, fontsize=15, tellwidth=false)
Label(fig[1, 0]; text="SIR",           font=:bold, fontsize=14, rotation=π/2, tellheight=false)
Label(fig[2, 0]; text="Beale",         font=:bold, fontsize=14, rotation=π/2, tellheight=false)
Label(fig[3, 0]; text="Goldstein-Price", font=:bold, fontsize=14, rotation=π/2, tellheight=false)

mkpath(PLOT_DIR)

for (idx, (ddir, prob, methods, title)) in enumerate(cells)
    r, c = positions[idx]
    @info "  [$r,$c] $prob ..."
    add_tv_panel!(fig[r, c], ddir, prob, methods;
        title  = title,
        legend = (idx == 1),
        ylabel = (c == 1),
    )
end

rowgap!(fig.layout, 12)
colgap!(fig.layout, 10)

save(joinpath(PLOT_DIR, "proxy_comparison_tv.png"), fig)
save(joinpath(PLOT_DIR, "proxy_comparison_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/proxy_comparison_tv.png"
