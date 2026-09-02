## Compare 3 setups on the 7 original Group A BIP problems, all at n=5 runs:
##   standard              — GP + MaxVar baseline               (data-bosip-norm/, run "standard", first 5 of 20 runs)
##   warpedgp-yja pre-fix   — WarpedGP YJ+Affine, BEFORE the BOSIP.jl predictive_samples fix
##                            (data-warpedgp2_archive_pre-samplesfix/, run "warpedgp-yja-maxvar")
##   warpedgp-yja post-fix  — WarpedGP YJ+Affine, AFTER the fix   (data-warpedgp2/, run "warpedgp-yja-maxvar")
##
## Purpose: visualize the effect of the predictive_samples fix (see project_bosip_warpedgp_moments_audit
## memory) against the standard-GP baseline, at matched n=5 sample size.
##
## Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR  = "plots"
const INIT_DATA = 3
mkpath(PLOT_DIR)

# ─────────────────────────────────────────────────────────
## Data loading (same convention as plot_warpedgp_all.jl)
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

## Fixed categorical assignment (matches plot_warpedgp_all.jl's convention where
## available): standard = WONG[2]/E69F00 (orange), post-fix WarpedGP = WONG[3]/009E73
## (bluish-green, same slot plot_warpedgp_all.jl uses for "WarpedGP YJ+Affine"),
## pre-fix (buggy) WarpedGP = WONG[6]/D55E00 (vermillion, signals "superseded/broken").
## Validated as a categorical triple (--pairs all, since all 3 co-occur across every
## small-multiple panel): CVD sep. worst 11.0 (pass, >=8 target), normal-vision floor
## 15.6 (pass, >=15), contrast — E69F00 at 2.19:1 is a WARN (relief), covered by the
## legend + direct per-line coloring already present. WONG[7]/F0E442 (yellow) was
## tried first and FAILS outright (L=0.90, contrast 1.29:1 — nearly invisible on white).
const METHODS = [
    MethodSpec("data-bosip-norm",                     "standard",             20, "Standard GP MaxVar (n=20)",            WONG[2]),
    MethodSpec("data-warpedgp2_archive_pre-samplesfix","warpedgp-yja-maxvar", 20, "WarpedGP YJ+Affine, pre-fix (n=20)",   WONG[6]),
    MethodSpec("data-warpedgp2",                       "warpedgp-yja-maxvar", 20, "WarpedGP YJ+Affine, post-fix (n=20)", WONG[3]),
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
            @warn "No data: $problem / $(m.run_name) in $(m.data_dir)"
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
## Figure: 7 original Group A BIP problems (2×4 grid, 1 empty slot)
# ─────────────────────────────────────────────────────────

const PROBLEMS = [
    ("ABProblem",          "AB"),
    ("SimpleProblem",      "Simple"),
    ("BananaProblem",      "Banana"),
    ("BimodalProblem",     "Bimodal"),
    ("SIRProblem",         "SIR"),
    ("DuffingProblem",     "Duffing"),
    ("DiffusionProblem10", "Diffusion"),
]

@info "=== WarpedGP samplesfix comparison (standard vs pre-fix vs post-fix) ==="

let
    ncols = 4
    ax_w, ax_h = 380, 300
    fig = Figure(; size=(ax_w * ncols + 220, ax_h * 2 + 60))

    positions = [(r, c) for r in 1:2 for c in 1:ncols]
    first_ax  = nothing

    for (idx, (pname, ptitle)) in enumerate(PROBLEMS)
        r, c = positions[idx]
        @info "  [$r,$c] $pname"
        ax = add_panel!(fig[r, c], pname, METHODS;
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
        path = joinpath(PLOT_DIR, "warpedgp_samplesfix_comparison.$ext")
        save(path, fig)
        @info "Saved $path"
    end
end

@info "Done."
