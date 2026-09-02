## Compare TV metric convergence across the 7 original BIP problems for 3 configs,
## using ONLY run #1 from each (same seed=555, same start_1.jld2 initial data for all three):
##   standard              — GP + LogMaxVar baseline           (data-bosip-norm/,      run 1)
##   warpedgp (old)         — WarpedGP YJ+Affine, pre-fix code  (data-warpedgp2-old/,   run 1)
##   warpedgp (new, fix)    — WarpedGP YJ+Affine, predictive_samples fix, LOCAL run     (data-warpedgp2/, run 1)
##
## Single-run comparison only — no aggregation/bands. The "new" line is dashed.
## Run via: julia --project=<local env> src/plot_warpedgp_fix_comparison_run1.jl
## (from the repo root, so relative data paths resolve correctly).

using JLD2
using CairoMakie

const PLOT_DIR  = "plots"
const INIT_DATA = 3
mkpath(PLOT_DIR)

function load_tv_score(data_dir::String, problem::String, run_name::String, idx::Int)
    fpath = joinpath(data_dir, problem, "$(run_name)_$(idx)_TVmetric.jld2")
    isfile(fpath) || return nothing
    s = load(fpath, "score")
    return isnothing(s) ? nothing : Float64.(s)
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

function add_line!(ax, s, color; linestyle=:solid, linewidth=2, label)
    isnothing(s) && return
    xs = collect(INIT_DATA .+ (0:length(s)-1))
    lines!(ax, xs, s; color, linewidth, linestyle, label)
end

function add_panel!(figpos, problem::String; title="", ylabel=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title,
        xscale = log10,
        yscale = log10,
    )

    std_s = load_tv_score("data-bosip-norm", problem, "standard", 1)
    isnothing(std_s) && @warn "No standard run1 data: $problem"
    add_line!(ax, std_s, WONG[2]; label="standard (run 1)")

    old_s = load_tv_score("data-warpedgp2-old", problem, "warpedgp-yja-maxvar", 1)
    isnothing(old_s) && @warn "No old-WarpedGP run1 data: $problem"
    add_line!(ax, old_s, WONG[6]; label="WarpedGP old (run 1)")

    new_s = load_tv_score("data-warpedgp2", problem, "warpedgp-yja-maxvar", 1)
    isnothing(new_s) && @warn "No new-WarpedGP run1 data: $problem"
    add_line!(ax, new_s, WONG[3]; linestyle=:dash, linewidth=2.5, label="WarpedGP NEW (fix, run 1)")

    return ax
end

@info "=== Plotting run #1 only: standard vs old-WarpedGP vs new-WarpedGP (fix) ==="

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
        "Single run per config (run #1) — same seed=555 & same initial data for all three, but each diverges from iteration 4 onward.";
        fontsize=13, color=:gray30, font=:italic)

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 8)
    rowsize!(fig.layout, 3, 30)

    for ext in ("pdf", "png")
        path = joinpath(PLOT_DIR, "warpedgp_fix_comparison_run1.$ext")
        save(path, fig)
        @info "Saved $path"
    end
end

@info "Done."
