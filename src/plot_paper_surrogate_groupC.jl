# Paper Plot 2c (appendix): Surrogate model comparison, Group C+D merged (all 6 HD
# problems: 2 HD BIP — DuffingProblem5, DiffusionProblem5D — + 4 HD cross-polytope
# opt-function problems — Rosenbrock5_cross, StyblinskiTang5_cross, Michalewicz5_cross,
# Sphere5_cross). Configs: baseline GP-MaxVar (`standard` for the BIP problems, `maxvar`
# for the cross-opt problems — same method, different run_name by historical
# convention), warpedgp-yja-maxvar, nongp.
#
# baseline (standard/maxvar) → data-bosip-norm/{Problem}/ (BIP) or
#                               data-opt-functions/{Problem}/ (cross-opt)
# warpedgp-yja-maxvar         → data-warpedgp2/{Problem}/ (both)
# nongp                       → same dir as baseline
#
# Data completeness (as of 2026-08-06, see reference_rci_cluster memory / project
# memory for full audit history):
#   baseline: 20/20 all 6 problems, clean (101 iters BIP, 201 iters cross-opt).
#   warpedgp-yja-maxvar: 20/20 all 6 problems, post predictive_samples-fix — clean
#     across the whole rerun (Groups B/C/D rerun completed 2026-07-27).
#   nongp: 20/20 files all 6 problems, but far short of the 200-iter target (NonstatGP
#     is expensive per-iteration at 5D) — accepted per iteration policy. DiffusionProblem5D
#     additionally has 6/20 runs crashed very early (PosDefException in
#     ConvergenceCallback, see project-bosip-benchmarks-nongp-expansion) — clips the
#     shared-valid-index nongp curve for that panel especially short; visible in the
#     curve itself, not auto-flagged (dq_problem_notes doesn't flag plain shortfalls).
#
# Layout: 3×2 grid (6 problems) + legend panel.
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

include("data_quality.jl")

const PLOT_DIR      = "plots"
const INIT_DATA     = 3
const BOSIPNORM_DIR = "data-bosip-norm"
const OPTFUN_DIR    = "data-opt-functions"
const WARPEDGP_DIR  = "data-warpedgp2"

# (display name, data dir for baseline/nongp, baseline run_name)
const PROBLEMS = [
    ("DuffingProblem5",              BOSIPNORM_DIR, "standard"),
    ("DiffusionProblem5D",           BOSIPNORM_DIR, "standard"),
    ("RosenbrockProblem5_cross",     OPTFUN_DIR,    "maxvar"),
    ("StyblinskiTangProblem5_cross", OPTFUN_DIR,    "maxvar"),
    ("MichalewiczProblem5_cross",    OPTFUN_DIR,    "maxvar"),
    ("SphereProblem5_cross",         OPTFUN_DIR,    "maxvar"),
]

# ─────────────────────────────────────────────
# Data routing
# ─────────────────────────────────────────────

function method_source(method::String, problem::String, base_dir::String, baseline::String)
    if method == "baseline"
        return base_dir, problem, baseline
    elseif method == "nongp"
        return base_dir, problem, "nongp"
    elseif method == "warpedgp-yja-maxvar"
        return WARPEDGP_DIR, problem, "warpedgp-yja-maxvar"
    end
    error("Unknown method: $method")
end

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

function load_tv_scores(data_dir::String, problem::String, run_name::String; max_runs::Int=20)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        fpath = joinpath(data_dir, problem, "$(run_name)_$(i)_TVmetric.jld2")
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
# Palette & labels (same slots as plot_paper_surrogate_merged.jl / groupB)
# ─────────────────────────────────────────────

const METHODS = ["baseline", "warpedgp-yja-maxvar", "nongp"]

## Canonical paper palette (2026-08-06, Wong/Okabe-Ito via Makie.wong_colors()):
## [1]=blue [2]=orange (reserved, niche custom-proxy plots) [3]=green (eiv,
## reserved even when eiv isn't in this figure) [4]=pink [5]=light/sky blue
## [6]=vermillion [7]=yellow
const PALETTE = Dict(
    "baseline"             => Makie.wong_colors()[2],   # orange (dark yellow)
    "warpedgp-yja-maxvar"  => Makie.wong_colors()[6],   # vermillion
    "nongp"                => Makie.wong_colors()[5],   # light/sky blue
)

const LABELS = Dict(
    "baseline"            => "GP - MaxVar",
    "warpedgp-yja-maxvar" => "WarpedGP - MaxVar",
    "nongp"               => "NonstatGP - MaxVar",
)

# ─────────────────────────────────────────────
# Per-panel plotting
# ─────────────────────────────────────────────

function add_tv_panel!(figpos, problem::String, base_dir::String, baseline::String; title="", ylabel=true, dq_acc=nothing)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        titlesize = 11,
        xscale = log10,
        yscale = log10,
    )

    for method in METHODS
        ddir, pname, run_name = method_source(method, problem, base_dir, baseline)
        scores = load_tv_scores(ddir, pname, run_name)
        isnothing(dq_acc) || dq_register!(dq_acc, problem, dq_stat(method, scores))
        if isempty(scores)
            @warn "No data for $problem / $method (looked in $ddir/$pname/$run_name)"
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

ncols = 3
nrows = 2
ax_w, ax_h = 280, 230

mkpath(PLOT_DIR)

fig = Figure(; size = (ax_w * ncols + 160, ax_h * nrows + 60))

ax_ref = nothing
DQ_ACC = Dict{String,Vector{Any}}()
for (idx, (pname, base_dir, baseline)) in enumerate(PROBLEMS)
    r = div(idx - 1, ncols) + 1
    c = mod(idx - 1, ncols) + 1
    @info "  [$r,$c] $pname ..."
    ax = add_tv_panel!(fig[r, c], pname, base_dir, baseline;
        title  = replace(replace(pname, "Problem" => ""), "_cross" => ""),
        ylabel = (c == 1),
        dq_acc = DQ_ACC,
    )
    if isnothing(ax_ref) && !isempty(ax.scene.plots)
        global ax_ref = ax
    end
end

if !isnothing(ax_ref)
    Legend(fig[1:nrows, ncols+1], ax_ref; orientation=:vertical, tellwidth=true, tellheight=false, labelsize=13, framevisible=true)
end

DQ_NOTES = String[]
for (p, s) in DQ_ACC
    append!(DQ_NOTES, dq_problem_notes(p, s))
end
sort!(DQ_NOTES)
dq_add_banner!(fig, DQ_NOTES; max_shown=8, fontsize=10)

rowgap!(fig.layout, 10)
colgap!(fig.layout, 8)

save(joinpath(PLOT_DIR, "paper_surrogate_groupC_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_surrogate_groupC_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_surrogate_groupC_tv.{png,pdf}"
