## Same 3-way comparison as plot_warpedgp_samplesfix_comparison.jl (standard vs
## pre-fix vs post-fix WarpedGP YJ+Affine, 7 original Group A BIP problems, n=20),
## but for the SIMULATOR-APPROXIMATION convergence metric (`_convergence.jld2`'s
## "score", from ConvergenceCallback/l2_norm — GP surrogate mean vs. true simulator
## output on a fixed reference grid) instead of the TV-distance posterior metric.
##
## NOTE: "standard" has no data here — those (older) runs predate script.jl's
## `convergence=true` default, so data-bosip-norm/ has no _convergence.jld2 files.
##
## NOTE: on SIR/Duffing/Diffusion10, this metric calls `mean(post,x)` directly
## (via ConvergenceCallback's sim_approx), which is moments-audit Category A — a
## code path NOT touched by the predictive_samples fix. It still overflows (Inf,
## or finite-but-astronomical up to ~1e32) on this metric's broad reference grid,
## roughly equally often pre- and post-fix (e.g. Duffing: 1609 vs 1767 Inf entries
## out of 2020) — confirming this is a separate, still-open numerical issue, not
## something the fix regressed. See project_bosip_warpedgp_moments_audit memory.
##
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

## ConvergenceCallback's "score" is a (n_dims, n_iters) matrix — l2_norm's per-dimension
## RMS errors. Combine dimensions via Euclidean norm (sqrt of sum of squares) to get one
## overall L2 error per iteration; a plain Vector (single-dim, older format) passes through.
## Inf entries occur when WarpedGP's `mean(post,x)` (used by sim_approx) overflows on a
## broad reference grid — a separate, unfixed code path (moments-audit Category A) from
## the predictive_samples fix. Treated as invalid, same as NaN, for plotting purposes.
function load_scores(data_dir::String, problem::String, run_name::String, suffix::String; max_runs::Int=20)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        fpath = joinpath(data_dir, problem, "$(run_name)_$(i)_$(suffix).jld2")
        isfile(fpath) || continue
        s = load(fpath, "score")
        isnothing(s) && continue
        combined = s isa AbstractMatrix ? vec(sqrt.(sum(abs2, s; dims=1))) : Float64.(s)
        replace!(x -> isinf(x) ? NaN : x, combined)
        push!(scores, combined)
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
## Method specs (identical assignment/colors to plot_warpedgp_samplesfix_comparison.jl —
## already validated there as a categorical triple; see that file for the CVD/contrast notes)
# ─────────────────────────────────────────────────────────

struct MethodSpec
    data_dir::String
    run_name::String
    max_runs::Int
    label::String
    color
end

const WONG = Makie.wong_colors()

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
        ylabel = ylabel ? "simulator L2 error" : "",
        title,
        xscale = log10,
        yscale = log10,
    )
    all_finite = Float64[]
    for m in methods
        scores = load_scores(m.data_dir, problem, m.run_name, "convergence"; max_runs=m.max_runs)
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
        append!(all_finite, filter(isfinite, med))
    end
    # A handful of individual runs (not the median/IQR shown) hit astronomical finite
    # values (up to ~1e32) from WarpedGP's mean(post,x) overflowing on the convergence
    # grid — a separate, unfixed numerical issue (see plot_warpedgp_samplesfix_convergence
    # header). Clip the view to the median traces' own range so those outliers (still
    # drawn, just off-canvas) don't blow out the axis; the underlying data is untouched.
    if !isempty(all_finite)
        lo_v, hi_v = extrema(all_finite)
        ylims!(ax, lo_v / 3, hi_v * 3)
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

@info "=== WarpedGP samplesfix comparison — simulator convergence (standard vs pre-fix vs post-fix) ==="

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
        path = joinpath(PLOT_DIR, "warpedgp_samplesfix_convergence.$ext")
        save(path, fig)
        @info "Saved $path"
    end
end

@info "Done."
