## Plot TV metric for all WarpedGP results vs standard GP baseline.
##
## 4 setups compared:
##   standard          — GP + MaxVar (data-bosip-norm/ or data-opt-functions/)
##   warpedgp-maxvar   — WarpedGP YJ+SA, fitted amplitude (data-warpedgp/)
##   warpedgp-yja-maxvar  — WarpedGP YJ+Affine, fixed amp=1 (data-warpedgp2/)
##   warpedgp-yjsa-maxvar — WarpedGP YJ+SA+Affine, fixed amp=1 (data-warpedgp2/)
##
## Figure 1: BIP problems  (2×4 grid, 8 panels)
## Figure 2: Opt problems  (1×4 grid, 4 panels: Beale, Goldstein, BealeProxy, GoldsteinProxy)
##
## Missing files are silently skipped (e.g. SIR yjsa still running).
## Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR  = "plots"
const INIT_DATA = 3
mkpath(PLOT_DIR)

# ─────────────────────────────────────────────────────────
## Data loading
# ─────────────────────────────────────────────────────────

function load_tv_scores(data_dir::String, problem::String, run_name::String; max_runs::Int=20)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        fpath = joinpath(data_dir, problem, "$(run_name)_$(i)_TVmetric.jld2")
        isfile(fpath) || continue
        s = load(fpath, "score")
        isnothing(s) || push!(scores, Float64.(s))
    end
    return scores
end

function aggregate(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] .= s
    end
    valid = [i for i in 1:maxlen if any(!isnan, mat[i, :])]
    isempty(valid) && return nothing
    xs  = collect(INIT_DATA .+ (valid .- 1))
    med = [median(filter(!isnan, mat[i, :])) for i in valid]
    lo  = [quantile(filter(!isnan, mat[i, :]), 0.25) for i in valid]
    hi  = [quantile(filter(!isnan, mat[i, :]), 0.75) for i in valid]
    return xs, med, lo, hi
end

# ─────────────────────────────────────────────────────────
## Method specs
# ─────────────────────────────────────────────────────────

struct MethodSpec
    data_dir::String
    run_name::String
    max_runs::Int
    label::String
    color
end

const WONG = Makie.wong_colors()

## BIP problems: standard GP in data-bosip-norm/ (20 runs, run_name "standard")
const BIP_METHODS = [
    MethodSpec("data-bosip-norm", "standard",           20, "GP MaxVar (n=20)",             WONG[2]),
    MethodSpec("data-warpedgp",   "warpedgp-maxvar",     5, "WarpedGP YJ+SA (n=5)",         WONG[6]),
    MethodSpec("data-warpedgp2",  "warpedgp-yja-maxvar", 5, "WarpedGP YJ+Affine (n=5)",     WONG[3]),
    MethodSpec("data-warpedgp2",  "warpedgp-yjsa-maxvar",5, "WarpedGP YJ+SA+Affine (n=5)",  WONG[5]),
]

## Opt-function problems: standard GP in data-opt-functions/ (5 runs, run_name "maxvar")
const OPT_METHODS = [
    MethodSpec("data-opt-functions", "maxvar",             5, "GP MaxVar (n=5)",              WONG[2]),
    MethodSpec("data-warpedgp",      "warpedgp-maxvar",    5, "WarpedGP YJ+SA (n=5)",        WONG[6]),
    MethodSpec("data-warpedgp2",     "warpedgp-yja-maxvar",5, "WarpedGP YJ+Affine (n=5)",    WONG[3]),
    MethodSpec("data-warpedgp2",     "warpedgp-yjsa-maxvar",5,"WarpedGP YJ+SA+Affine (n=5)", WONG[5]),
]

# ─────────────────────────────────────────────────────────
## Panel rendering
# ─────────────────────────────────────────────────────────

function add_panel!(figpos, problem::String, methods::Vector{MethodSpec};
                    title="", ylabel=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title,
        xscale = log10,
        yscale = log10,
    )
    for m in methods
        scores = load_tv_scores(m.data_dir, problem, m.run_name; max_runs=m.max_runs)
        if isempty(scores)
            @warn "No data: $problem / $(m.run_name)"
            continue
        end
        agg = aggregate(scores)
        isnothing(agg) && continue
        xs, med, lo, hi = agg
        c = m.color
        n = length(scores)

        for s in scores
            xs_s = collect(INIT_DATA .+ (0:length(s)-1))
            lines!(ax, xs_s, s; color=(c, 0.15), linewidth=0.7)
        end
        n > 1 && band!(ax, xs, lo, hi; color=(c, 0.20))
        lines!(ax, xs, med; color=c, linewidth=2, label=m.label)
    end
    return ax
end

# ─────────────────────────────────────────────────────────
## Figure 1: BIP problems (2×4)
# ─────────────────────────────────────────────────────────

const BIP_PROBLEMS = [
    ("ABProblem",          "AB"),
    ("SimpleProblem",      "Simple"),
    ("BananaProblem",      "Banana"),
    ("BimodalProblem",     "Bimodal"),
    ("SIRProblem",         "SIR"),
    ("ProxySIRProblem",    "Proxy SIR"),
    ("DuffingProblem",     "Duffing"),
    ("DiffusionProblem10", "Diffusion"),
]

@info "=== Figure 1: BIP problems ==="

let
    ncols = 4
    ax_w, ax_h = 380, 300
    fig = Figure(; size=(ax_w * ncols + 220, ax_h * 2 + 60))

    positions = [(r, c) for r in 1:2 for c in 1:ncols]
    first_ax  = nothing

    for (idx, (pname, ptitle)) in enumerate(BIP_PROBLEMS)
        r, c = positions[idx]
        @info "  [$r,$c] $pname"
        ax = add_panel!(fig[r, c], pname, BIP_METHODS;
            title  = ptitle,
            ylabel = (c == 1),
        )
        isnothing(first_ax) && (first_ax = ax)
    end

    if !isnothing(first_ax)
        Legend(fig[1:2, ncols+1], first_ax;
            tellwidth=true, labelsize=13, framevisible=true,
            title="Method")
    end
    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 8)

    for ext in ("pdf", "png")
        path = joinpath(PLOT_DIR, "warpedgp_all_bip.$ext")
        save(path, fig)
        @info "Saved $path"
    end
end

# ─────────────────────────────────────────────────────────
## Figure 2: Opt-function problems (1×4)
# ─────────────────────────────────────────────────────────

const OPT_PROBLEMS = [
    ("BealeProblem",               "Beale"),
    ("GoldsteinPriceProblem",      "Goldstein-Price"),
    ("BealeProxyProblem",          "Beale (proxy)"),
    ("GoldsteinPriceProxyProblem", "Goldstein-Price (proxy)"),
]

@info "=== Figure 2: Opt-function problems ==="

let
    ax_w, ax_h = 380, 300
    fig = Figure(; size=(ax_w * 4 + 220, ax_h + 60))

    first_ax = nothing
    for (col, (pname, ptitle)) in enumerate(OPT_PROBLEMS)
        @info "  [1,$col] $pname"
        ax = add_panel!(fig[1, col], pname, OPT_METHODS;
            title  = ptitle,
            ylabel = (col == 1),
        )
        isnothing(first_ax) && (first_ax = ax)
    end

    if !isnothing(first_ax)
        Legend(fig[1, 5], first_ax;
            tellwidth=true, labelsize=13, framevisible=true,
            title="Method")
    end
    colgap!(fig.layout, 8)

    for ext in ("pdf", "png")
        path = joinpath(PLOT_DIR, "warpedgp_all_opt.$ext")
        save(path, fig)
        @info "Saved $path"
    end
end

@info "Done."
