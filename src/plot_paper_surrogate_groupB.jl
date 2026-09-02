# Paper Plot 2b (appendix): Surrogate model comparison, Group B (24 cross-polytope
# 2D opt-function problems). Configs: maxvar (=standard baseline), warpedgp-yja-maxvar,
# nongp.
#
# maxvar               → data-opt-functions/{Problem}_cross/    (GP, MaxVar baseline)
# warpedgp-yja-maxvar  → data-warpedgp2/{Problem}_cross/         (WarpedGP YJ+Affine, LogMaxVar)
# nongp                → data-opt-functions/{Problem}_cross/     (NonstatGP, MaxVar)
#
# Data completeness (as of 2026-08-03, see cluster_scripts/notes_paper_plots.md):
#   maxvar: 20/20 all 24 problems, 201 iters, clean.
#   warpedgp-yja-maxvar: 20/20 all 24 problems, 201 iters, 0 NaN (fully resolved since
#     the 2026-07-24 partial-coverage audit — all previously-missing/broken problems
#     (DropWave, Easom, GoldsteinPrice, Himmelblau, HolderTable, LeviN13, Matyas,
#     SchafferN2, ThreeHumpCamel) now complete and clean).
#   nongp: 20/20 files all 24 problems but far short of the 100-iter target
#     (39-71 iters depending on problem, per the 2026-08-01 nongp-expansion audit) with
#     heavy NaN contamination on several problems (Beale 492, GoldsteinPrice 466,
#     Himmelblau 242, Booth 172, LeviN13 158, Rosenbrock2 152, Sphere2 128 — out of
#     ~1000-1400 total readings per problem). Accepted per iteration policy (nongp is
#     expensive per-iteration; not resubmitted — plotting-only pass).
#
# Layout: 4×6 grid (24 problems) + legend panel.
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

include("data_quality.jl")

const PLOT_DIR      = "plots"
const INIT_DATA     = 3
const OPTFUN_DIR    = "data-opt-functions"
const WARPEDGP_DIR  = "data-warpedgp2"

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
# Data routing
# ─────────────────────────────────────────────

function method_source(method::String, problem_cross::String)
    if method in ("maxvar", "nongp")
        return OPTFUN_DIR, problem_cross
    elseif method == "warpedgp-yja-maxvar"
        return WARPEDGP_DIR, problem_cross
    end
    error("Unknown method: $method")
end

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
# Palette & labels (same slots as plot_paper_surrogate_merged.jl)
# ─────────────────────────────────────────────

const METHODS = ["maxvar", "warpedgp-yja-maxvar", "nongp"]

## Canonical paper palette (2026-08-06, Wong/Okabe-Ito via Makie.wong_colors()):
## [1]=blue [2]=orange (reserved, niche custom-proxy plots) [3]=green (eiv,
## reserved even when eiv isn't in this figure) [4]=pink [5]=light/sky blue
## [6]=vermillion [7]=yellow
const PALETTE = Dict(
    "maxvar"               => Makie.wong_colors()[2],   # orange (dark yellow)
    "warpedgp-yja-maxvar"  => Makie.wong_colors()[6],   # vermillion
    "nongp"                => Makie.wong_colors()[5],   # light/sky blue
)

const LABELS = Dict(
    "maxvar"              => "GP - MaxVar",
    "warpedgp-yja-maxvar" => "WarpedGP - LogMaxVar",
    "nongp"               => "NonstatGP - MaxVar",
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
        ddir, pname = method_source(method, problem_cross)
        scores = load_tv_scores(ddir, pname, method)
        isnothing(dq_acc) || dq_register!(dq_acc, problem_cross, dq_stat(method, scores))
        if isempty(scores)
            @warn "No data for $problem_cross / $method (looked in $ddir/$pname)"
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

save(joinpath(PLOT_DIR, "paper_surrogate_groupB_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_surrogate_groupB_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_surrogate_groupB_tv.{png,pdf}"
