## WarpedGP comparison plots
##
## Plot 1 — GP vs WarpedGP vs NonstatGP, all with MaxVar
##   3 panels: ABProblem, SIRProblem, ProxySIRProblem
##   Saved to plots/warpedgp_maxvar_comparison.pdf
##
## Plot 2 — GP vs WarpedGP with MaxVar and EIV
##   3 panels: ABProblem, SIRProblem, ProxySIRProblem
##   Each panel shows 4 curves: GP-MaxVar, GP-EIV, WarpedGP-MaxVar, WarpedGP-EIV
##   Saved to plots/warpedgp_eiv_comparison.pdf
##
## Baseline data:  data-bosip-norm/  (run_names: standard, eiv, nongp)
## WarpedGP data:  data-warpedgp/    (run_names: warpedgp-maxvar, warpedgp-eiv)
## Missing files are silently skipped — partial SIR eiv data plots as a shorter curve.
##
## Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR  = "plots"
const INIT_DATA = 3

# ─────────────────────────────────────────────────────────
## Data loading
# ─────────────────────────────────────────────────────────

function load_tv_scores(data_dir::String, problem::String, run_name::String; max_runs::Int=10)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        fpath = joinpath(data_dir, problem, "$(run_name)_$(i)_TVmetric.jld2")
        isfile(fpath) || continue
        s = load(fpath, "score")
        isnothing(s) || push!(scores, Float64.(s))
    end
    return scores
end

function median_and_band(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] .= s
    end
    valid = [i for i in 1:maxlen if any(!isnan, mat[i, :])]
    isempty(valid) && return nothing
    xs  = INIT_DATA .+ (valid .- 1)
    med = [median(filter(!isnan, mat[i, :])) for i in valid]
    lo  = [quantile(filter(!isnan, mat[i, :]), 0.25) for i in valid]
    hi  = [quantile(filter(!isnan, mat[i, :]), 0.75) for i in valid]
    return collect(xs), med, lo, hi
end

# ─────────────────────────────────────────────────────────
## Method specs
# ─────────────────────────────────────────────────────────

struct MethodSpec
    data_dir::String
    run_name::String
    label::String
    color
    linestyle::Symbol
end

const BOSIP_NORM = "data-bosip-norm"
const WARPED_GP  = "data-warpedgp"

const WONG = Makie.wong_colors()

## Plot 1 methods: GP-MaxVar, WarpedGP-MaxVar, NonstatGP-MaxVar
const METHODS_MAXVAR = [
    MethodSpec(BOSIP_NORM, "standard",       "GP (MaxVar)",       WONG[2], :solid),
    MethodSpec(WARPED_GP,  "warpedgp-maxvar","WarpedGP (MaxVar)", WONG[6], :solid),
    MethodSpec(BOSIP_NORM, "nongp",          "NonstatGP (MaxVar)",WONG[3], :solid),
]

## Plot 2 methods: GP-MaxVar, GP-EIV, WarpedGP-MaxVar, WarpedGP-EIV
const METHODS_GP_VS_WARPEDGP = [
    MethodSpec(BOSIP_NORM, "standard",       "GP (MaxVar)",       WONG[2], :solid),
    MethodSpec(BOSIP_NORM, "eiv",            "GP (EIV)",          WONG[1], :solid),
    MethodSpec(WARPED_GP,  "warpedgp-maxvar","WarpedGP (MaxVar)", WONG[6], :dash),
    MethodSpec(WARPED_GP,  "warpedgp-eiv",   "WarpedGP (EIV)",   WONG[5], :dash),
]

const PROBLEMS = [
    ("ABProblem",       "AB Problem"),
    ("SIRProblem",      "SIR Problem"),
    ("ProxySIRProblem", "Proxy SIR"),
]

# ─────────────────────────────────────────────────────────
## Panel rendering
# ─────────────────────────────────────────────────────────

function add_panel!(figpos, problem::String, methods::Vector{MethodSpec}; title="", ylabel=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        xscale = log10,
        yscale = log10,
    )

    for m in methods
        scores = load_tv_scores(m.data_dir, problem, m.run_name)
        if isempty(scores)
            @warn "No data: $problem / $(m.run_name)"
            continue
        end
        agg = median_and_band(scores)
        isnothing(agg) && continue
        xs, med, lo, hi = agg
        n = length(scores)
        c = m.color

        for s in scores
            xs_s = collect(INIT_DATA .+ (0:length(s)-1))
            lines!(ax, xs_s, s; color=(c, 0.12), linewidth=0.6)
        end
        band!(ax, xs, lo, hi; color=(c, 0.20))
        lines!(ax, xs, med; color=c, linewidth=2, linestyle=m.linestyle,
               label="$(m.label) (n=$n)")
    end

    return ax
end

# ─────────────────────────────────────────────────────────
## Build and save figures
# ─────────────────────────────────────────────────────────

mkpath(PLOT_DIR)

function make_figure(methods::Vector{MethodSpec}, outname::String)
    ax_w, ax_h = 400, 320
    fig = Figure(; size=(ax_w * length(PROBLEMS) + 80, ax_h + 80))

    local first_ax = nothing
    for (col, (pname, ptitle)) in enumerate(PROBLEMS)
        @info "  [$pname] $(outname) ..."
        ax = add_panel!(fig[1, col], pname, methods;
            title  = ptitle,
            ylabel = (col == 1),
        )
        if isnothing(first_ax)
            first_ax = ax
        end
    end

    if !isnothing(first_ax)
        Legend(fig[1, length(PROBLEMS)+1], first_ax;
            tellwidth=false, tellheight=false, labelsize=13, framevisible=true)
    end

    colgap!(fig.layout, 10)

    for ext in ("pdf", "png")
        path = joinpath(PLOT_DIR, "$outname.$ext")
        save(path, fig)
        @info "Saved $path"
    end
    return fig
end

@info "=== Plot 1: MaxVar comparison (GP / WarpedGP / NonstatGP) ==="
make_figure(METHODS_MAXVAR, "warpedgp_maxvar_comparison")

@info "=== Plot 2: GP vs WarpedGP (MaxVar + EIV) ==="
make_figure(METHODS_GP_VS_WARPEDGP, "warpedgp_eiv_comparison")

@info "Done."
