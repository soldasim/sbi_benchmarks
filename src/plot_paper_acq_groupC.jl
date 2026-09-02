# Paper Plot 3c (appendix): Acquisition function comparison, Group C+D merged (all 6
# HD problems: 2 HD BIP — DuffingProblem5, DiffusionProblem5D — + 4 HD cross-polytope
# opt-function problems — Rosenbrock5_cross, StyblinskiTang5_cross, Michalewicz5_cross,
# Sphere5_cross). Configs: baseline (=MaxVar, `standard` for BIP / `maxvar` for
# cross-opt), eiv, immd. (No eiig — only ever run for Group A.)
#
# baseline/eiv/immd → data-bosip-norm/{Problem}/ (BIP) or data-opt-functions/{Problem}/
# (cross-opt) — same dir for all 3 methods within a problem.
#
# Data completeness (as of 2026-08-06, see reference_rci_cluster memory / project
# memory for full audit history):
#   baseline: 20/20 all 6 problems.
#   eiv/immd: only 5/20 runs each (Groups C/D policy: EIV/IMMD only get runs 1-5 at
#     HD), both timed out well short of the 200/201-iter target — accepted per
#     iteration policy, no resubmission, plotting-only pass.
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

# (display name, data dir, baseline run_name)
const PROBLEMS = [
    ("DuffingProblem5",              BOSIPNORM_DIR, "standard"),
    ("DiffusionProblem5D",           BOSIPNORM_DIR, "standard"),
    ("RosenbrockProblem5_cross",     OPTFUN_DIR,    "maxvar"),
    ("StyblinskiTangProblem5_cross", OPTFUN_DIR,    "maxvar"),
    ("MichalewiczProblem5_cross",    OPTFUN_DIR,    "maxvar"),
    ("SphereProblem5_cross",         OPTFUN_DIR,    "maxvar"),
]

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
# Palette & labels (same slots as plot_paper_acq.jl / groupB)
# ─────────────────────────────────────────────

const METHODS = ["baseline", "eiv", "immd"]

## Canonical paper palette (2026-08-06, Wong/Okabe-Ito via Makie.wong_colors()):
## [1]=blue [2]=orange (reserved, niche custom-proxy plots) [3]=green [4]=pink
## [5]=light/sky blue [6]=vermillion [7]=yellow
const PALETTE = Dict(
    "baseline" => Makie.wong_colors()[2],   # orange (dark yellow)
    "eiv"      => Makie.wong_colors()[3],   # green
    "immd"     => Makie.wong_colors()[4],   # pink
)

const LABELS = Dict(
    "baseline" => "MaxVar",
    "eiv"      => "EIV",
    "immd"     => "IMMD",
)

# ─────────────────────────────────────────────
# Per-panel plotting
# ─────────────────────────────────────────────

function add_tv_panel!(figpos, problem::String, data_dir::String, baseline::String; title="", ylabel=true, dq_acc=nothing)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        titlesize = 11,
        xscale = log10,
        yscale = log10,
    )

    for method in METHODS
        run_name = method == "baseline" ? baseline : method
        scores = load_tv_scores(data_dir, problem, run_name)
        isnothing(dq_acc) || dq_register!(dq_acc, problem, dq_stat(method, scores; expected=(method == "baseline" ? 20 : 5)))
        if isempty(scores)
            @warn "No data for $problem / $method (run_name=$run_name)"
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
for (idx, (pname, data_dir, baseline)) in enumerate(PROBLEMS)
    r = div(idx - 1, ncols) + 1
    c = mod(idx - 1, ncols) + 1
    @info "  [$r,$c] $pname ..."
    ax = add_tv_panel!(fig[r, c], pname, data_dir, baseline;
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

save(joinpath(PLOT_DIR, "paper_acq_groupC_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_acq_groupC_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_acq_groupC_tv.{png,pdf}"
