## Compare TV metric convergence across the 7 original BIP problems for 3 configs:
##   standard              — GP + LogMaxVar baseline           (data-bosip-norm/,      n=20)
##   warpedgp (old)         — WarpedGP YJ+Affine, pre-fix code  (data-warpedgp2-old/,   n=20)
##   warpedgp (new, fix)    — WarpedGP YJ+Affine, predictive_samples fix, LOCAL run (data-warpedgp2/, n=1)
##
## The "new" line is dashed and annotated as a single preliminary run — NOT yet a fair
## n=20-vs-n=20 comparison. Run via: julia --project=<local env> src/plot_warpedgp_fix_comparison.jl
## (from the repo root, so relative data paths resolve correctly).

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR  = "plots"
const INIT_DATA = 3
mkpath(PLOT_DIR)

function load_tv_scores(data_dir::String, problem::String, run_name::String, indices)
    scores = Vector{Float64}[]
    for i in indices
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

const WONG = Makie.wong_colors()

const PROBLEMS = [
    ("ABProblem",          "AB"),
    ("SimpleProblem",      "Simple"),
    ("BananaProblem",      "Banana"),
    ("BimodalProblem",     "Bimodal"),
    ("SIRProblem",         "SIR"),
    ("DuffingProblem",     "Duffing"),
    ("DiffusionProblem10", "Diffusion"),
]

function add_panel!(figpos, problem::String; title="", ylabel=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title,
        xscale = log10,
        yscale = log10,
    )

    # standard baseline (n=20, solid)
    std_scores = load_tv_scores("data-bosip-norm", problem, "standard", 1:20)
    if !isempty(std_scores)
        agg = aggregate(std_scores)
        if !isnothing(agg)
            xs, med, lo, hi = agg
            band!(ax, xs, lo, hi; color=(WONG[2], 0.20))
            lines!(ax, xs, med; color=WONG[2], linewidth=2, label="standard (n=$(length(std_scores)))")
        end
    else
        @warn "No standard data: $problem"
    end

    # old WarpedGP (pre-fix, n=20, solid)
    old_scores = load_tv_scores("data-warpedgp2-old", problem, "warpedgp-yja-maxvar", 1:20)
    if !isempty(old_scores)
        agg = aggregate(old_scores)
        if !isnothing(agg)
            xs, med, lo, hi = agg
            band!(ax, xs, lo, hi; color=(WONG[6], 0.20))
            lines!(ax, xs, med; color=WONG[6], linewidth=2, label="WarpedGP old (n=$(length(old_scores)))")
        end
    else
        @warn "No old-WarpedGP data: $problem"
    end

    # new WarpedGP (post-fix, n=1, dashed — preliminary)
    new_scores = load_tv_scores("data-warpedgp2", problem, "warpedgp-yja-maxvar", 1:1)
    if !isempty(new_scores)
        s = new_scores[1]
        xs_s = collect(INIT_DATA .+ (0:length(s)-1))
        lines!(ax, xs_s, s; color=WONG[3], linewidth=2.5, linestyle=:dash,
            label="WarpedGP NEW (fix, n=1, preliminary)")
    else
        @warn "No new-WarpedGP data: $problem"
    end

    return ax
end

@info "=== Plotting standard vs old-WarpedGP vs new-WarpedGP (fix) ==="

let
    ncols = 4
    ax_w, ax_h = 380, 300
    fig = Figure(; size=(ax_w * ncols + 260, ax_h * 2 + 90))

    positions = [(r, c) for r in 1:2 for c in 1:ncols]
    first_ax  = nothing

    for (idx, (pname, ptitle)) in enumerate(PROBLEMS)
        r, c = positions[idx]
        @info "  [$r,$c] $pname"
        ax = add_panel!(fig[r, c], pname; title=ptitle, ylabel=(c == 1))
        isnothing(first_ax) && (first_ax = ax)
    end

    if !isnothing(first_ax)
        Legend(fig[1:2, ncols+1], first_ax;
            tellwidth=true, labelsize=13, framevisible=true,
            title="Method")
    end

    Label(fig[3, 1:ncols],
        "⚠ \"WarpedGP NEW\" (dashed) is a SINGLE run per problem — preliminary, not yet a fair n-vs-n comparison.";
        fontsize=14, color=:red, font=:bold_italic)

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 8)
    rowsize!(fig.layout, 3, 30)

    for ext in ("pdf", "png")
        path = joinpath(PLOT_DIR, "warpedgp_fix_comparison.$ext")
        save(path, fig)
        @info "Saved $path"
    end
end

@info "Done."
