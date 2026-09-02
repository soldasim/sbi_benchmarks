# Paper Plot 3b (appendix): Acquisition function comparison, Group B (24
# cross-polytope 2D opt-function problems). Configs: maxvar (=standard baseline),
# eiv, immd. (No eiig for Group B — only ever run for Group A.)
#
# All three configs live in data-opt-functions/{Problem}_cross/.
#
# Data completeness (as of 2026-08-03, see cluster_scripts/notes_paper_plots.md):
#   maxvar: 20/20 all 24, 201 iters. Known NaN: BealeProblem_cross (405 total, a
#     recurring/fix-resistant failure per 2026-07-28 investigation, accepted as-is),
#     GoldsteinPriceProblem_cross (53 total, small residual). All other 22 clean.
#   eiv: all 24 reach the 101-iter target (some further, timed-out extensions to
#     ~201). NaN: BealeProblem_cross (206, same accepted caveat as maxvar),
#     GoldsteinPriceProblem_cross (21, small residual). All other 22 problems 0 NaN
#     (the 6-problem NaN cluster from the 2026-07-24 audit — Rosenbrock2,
#     StyblinskiTang2, Schwefel2, Himmelblau, HolderTable, Booth — was fully resolved
#     via TV-only recompute on 2026-08-03).
#   immd: clean except SchwefelProblem2_cross (52-101/101 iters, known 24h-timeout
#     gap, 0 NaN, accepted).
#
# Layout: 4×6 grid (24 problems) + legend panel.
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

include("data_quality.jl")

const PLOT_DIR   = "plots"
const INIT_DATA  = 3
const OPTFUN_DIR = "data-opt-functions"

const PROBLEMS = [
    "RosenbrockProblem2", "StyblinskiTangProblem2", "MichalewiczProblem2",
    "AckleyProblem2", "AlpineProblem2", "ExpandedSchafferF6Problem2",
    "ExpandedZakharovProblem2", "GriewankProblem2", "RastriginProblem2",
    "SalomonProblem2", "SchwefelProblem2", "SphereProblem2",
    "BealeProblem", "BoothProblem", "CrossInTrayProblem",
    "DropWaveProblem", "EasomProblem", "GoldsteinPriceProblem",
    "HimmelblauProblem", "HolderTableProblem", "LeviN13Problem",
    "MatyasProblem", "SchafferN2Problem", "ThreeHumpCamelProblem",
]

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

function load_tv_scores(data_dir::String, problem::String, method::String; max_runs::Int=20)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        fpath = joinpath(data_dir, problem, "$(method)_$(i)_TVmetric.jld2")
        isfile(fpath) || continue
        s = load(fpath, "score")
        isnothing(s) || push!(scores, s)
    end
    return scores
end

function median_and_band(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] = s
    end
    valid = [i for i in 1:maxlen if all(!isnan, mat[i, :])]
    isempty(valid) && return nothing
    xs  = INIT_DATA .+ (valid .- 1)
    med = [median(filter(!isnan, mat[i, :])) for i in valid]
    lo  = [quantile(filter(!isnan, mat[i, :]), 0.25) for i in valid]
    hi  = [quantile(filter(!isnan, mat[i, :]), 0.75) for i in valid]
    return collect(xs), med, lo, hi
end

# ─────────────────────────────────────────────
# Palette & labels (same slots as plot_paper_acq.jl)
# ─────────────────────────────────────────────

const METHODS = ["maxvar", "eiv", "immd"]

## Canonical paper palette (2026-08-06, Wong/Okabe-Ito via Makie.wong_colors()):
## [1]=blue [2]=orange (reserved, niche custom-proxy plots) [3]=green [4]=pink
## [5]=light/sky blue [6]=vermillion [7]=yellow
const PALETTE = Dict(
    "maxvar" => Makie.wong_colors()[2],   # orange (dark yellow)
    "eiv"    => Makie.wong_colors()[3],   # green
    "immd"   => Makie.wong_colors()[4],   # pink
)

const LABELS = Dict(
    "maxvar" => "MaxVar",
    "eiv"    => "EIV",
    "immd"   => "IMMD",
)

# ─────────────────────────────────────────────
# Per-panel plotting
# ─────────────────────────────────────────────

function add_tv_panel!(figpos, problem_cross::String; title="", ylabel=true, dq_acc=nothing)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        titlesize = 11,
        xscale = log10,
        yscale = log10,
    )

    for method in METHODS
        scores = load_tv_scores(OPTFUN_DIR, problem_cross, method)
        isnothing(dq_acc) || dq_register!(dq_acc, problem_cross, dq_stat(method, scores))
        if isempty(scores)
            @warn "No data for $problem_cross / $method"
            continue
        end

        col = PALETTE[method]
        lbl = LABELS[method]
        agg = median_and_band(scores)
        isnothing(agg) && continue
        xs_med, med, lo, hi = agg

        lines!(ax, xs_med, med; color=col, linewidth=1.5, label="$lbl")
    end

    return ax
end

# ─────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────

ncols = 6
nrows = 4
ax_w, ax_h = 260, 210

mkpath(PLOT_DIR)

fig = Figure(; size = (ax_w * ncols + 40, ax_h * nrows + 100))

ax_ref = nothing
DQ_ACC = Dict{String,Vector{Any}}()
for (idx, pname) in enumerate(PROBLEMS)
    r = div(idx - 1, ncols) + 1
    c = mod(idx - 1, ncols) + 1
    problem_cross = pname * "_cross"
    @info "  [$r,$c] $problem_cross ..."
    ax = add_tv_panel!(fig[r, c], problem_cross;
        title  = replace(pname, "Problem" => ""),
        ylabel = (c == 1),
        dq_acc = DQ_ACC,
    )
    if isnothing(ax_ref) && !isempty(ax.scene.plots)
        global ax_ref = ax
    end
end

if !isnothing(ax_ref)
    Legend(fig[nrows+1, :], ax_ref; orientation=:horizontal, tellwidth=false, tellheight=true, labelsize=13, framevisible=true)
end

DQ_NOTES = String[]
for (p, s) in DQ_ACC
    append!(DQ_NOTES, dq_problem_notes(p, s))
end
sort!(DQ_NOTES)
dq_add_banner!(fig, DQ_NOTES; max_shown=8, fontsize=10)

rowgap!(fig.layout, 10)
colgap!(fig.layout, 8)

save(joinpath(PLOT_DIR, "paper_acq_groupB_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_acq_groupB_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_acq_groupB_tv.{png,pdf}"
