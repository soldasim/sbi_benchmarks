# Paper Plot 2a: Surrogate model comparison (Group A, 7 orig BIP problems).
# Merges the two former separate plots (standard-vs-warpedgp, standard-vs-nongp)
# into one figure per the 2026-07-15 plan renumbering.
#
# standard              → data-bosip-norm/{Problem}/               (GP, MaxVar)
# warpedgp-yja-maxvar   → data-warpedgp2/{Problem}/                 (WarpedGP YJ+Affine, LogMaxVar; post predictive_samples fix)
# nongp                 → data-bosip-norm/{Problem}/                (NonstationaryGP, MaxVar)
#
# Data completeness (as of 2026-08-06, see cluster_scripts/notes_paper_plots.md):
#   standard: 20/20 all 7 problems, clean, target reached.
#   warpedgp-yja-maxvar: 20/20 all 7 problems, clean (0 NaN), target reached (post-fix).
#   nongp: SEVERELY short of the 101-iter target everywhere (30-82 iters depending on
#     problem) with mild NaN on Simple (142 total) and SIR (53 total). Accepted as final
#     per iteration policy (NonstatGP fit cost scales badly with data size).
#     Per-problem nongp iter range: AB 39-57, Simple 78-82, Banana 35-59, Bimodal 34-36,
#     SIR 45-63 (see below), Duffing 33-36, Diffusion10 30-31.
#     - DuffingProblem: DO NOT use `data-bosip/DuffingProblem/nongp_*` — investigated
#       2026-08-06 and found it holds scores in the THOUSANDS (not [0,1]), because a
#       2026-07-24 continuation job ran through an override script tied to the old,
#       pre-2026-07-14-TV-normalization-fix code path and silently recomputed the
#       *entire* array (not just new iterations) under the broken normalization.
#       `data-bosip-norm/DuffingProblem/nongp_*` (33-36 iters, 0 NaN, values in [0,1])
#       remains the only valid copy — stays in use here. Properly extending this data
#       would need a fresh TV-metric recompute under the corrected normalization
#       applied to the longer `data-bosip` data.jld2 (like the AB/Banana/Bimodal/
#       Diffusion10 recompute did) — not attempted here, separate future work.
#     - SIRProblem: runs 1-2 are pre-fix stragglers (only 31 iters, vs. runs 3-20's
#       post-fix 45-63) — a rerun was submitted 2026-08-06 (jobs 11313592/11313593,
#       ~3-day cpulong, not done as of this plot). Runs 1-2 are EXCLUDED below until
#       that rerun lands — see `SIR_NONGP_EXCLUDE` and the manual caveat banner in the
#       figure. Re-run this script once the rerun completes, then remove the exclusion.
#
# Layout: 2×4 grid (same as plot_paper_proxy.jl / plot_paper_acq.jl)
#   Row 1: AB, Simple, Banana, Bimodal
#   Row 2: SIR, Duffing, Diffusion, (legend)
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

include("data_quality.jl")

const PLOT_DIR    = "plots"
const INIT_DATA   = 3
const NORM_DIR    = "data-bosip-norm"
const WARPEDGP_DIR = "data-warpedgp2"

# Per-problem iteration target: 100 for the 2D problems, 200 for DuffingProblem
# (the one 3D problem in this group — harder, so it gets the higher HD-style
# target). Enforced explicitly here rather than relying on whichever run happens
# to be shortest on disk.
target_iters(problem_name::String) = problem_name == "DuffingProblem" ? 200 : 100

# ─────────────────────────────────────────────
# Data routing: (data_dir, problem_name) per method
# ─────────────────────────────────────────────

function method_source(method::String, problem::String)
    if method in ("standard", "nongp")
        return NORM_DIR, problem
    elseif method == "warpedgp-yja-maxvar"
        return WARPEDGP_DIR, problem
    end
    error("Unknown method: $method")
end

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

function load_tv_scores(data_dir::String, problem::String, method::String;
                         max_runs::Int=20, exclude::Set{Int}=Set{Int}())
    scores = Vector{Float64}[]
    for i in 1:max_runs
        i in exclude && continue
        fpath = joinpath(data_dir, problem, "$(method)_$(i)_TVmetric.jld2")
        isfile(fpath) || continue
        s = load(fpath, "score")
        isnothing(s) || push!(scores, s)
    end
    return scores
end

# TEMPORARY (as of 2026-08-06): SIRProblem nongp runs 1-2 are mid-rerun (jobs
# 11313592/11313593, fixing a pre-fix straggler — see header note above). Exclude
# them from this plot until that rerun completes; remove this once it lands.
const SIR_NONGP_EXCLUDE = Set([1, 2])

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
# Palette & labels
# ─────────────────────────────────────────────

const METHODS = ["standard", "warpedgp-yja-maxvar", "nongp"]

## Canonical paper palette (2026-08-06, Wong/Okabe-Ito via Makie.wong_colors()):
## [1]=blue [2]=orange (reserved, niche custom-proxy plots) [3]=green (eiv,
## reserved even when eiv isn't in this figure) [4]=pink [5]=light/sky blue
## [6]=vermillion [7]=yellow
const PALETTE = Dict(
    "standard"             => Makie.wong_colors()[2],   # orange (dark yellow) — GP / MaxVar
    "warpedgp-yja-maxvar"  => Makie.wong_colors()[6],   # vermillion — WarpedGP / LogMaxVar
    "nongp"                => Makie.wong_colors()[5],   # light/sky blue — NonstatGP / MaxVar
)

const LABELS = Dict(
    "standard"            => "GP - MaxVar",
    "warpedgp-yja-maxvar" => "WarpedGP - LogMaxVar",
    "nongp"               => "NonstatGP - MaxVar",
)

# ─────────────────────────────────────────────
# Per-panel plotting
# ─────────────────────────────────────────────

function add_tv_panel!(figpos, problem_name::String; title="", ylabel=true, dq_acc=nothing)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        xscale = log10,
        yscale = log10,
    )

    for method in METHODS
        ddir, pname = method_source(method, problem_name)
        exclude = (method == "nongp" && problem_name == "SIRProblem") ? SIR_NONGP_EXCLUDE : Set{Int}()
        scores = load_tv_scores(ddir, pname, method; exclude=exclude)
        isnothing(dq_acc) || dq_register!(dq_acc, problem_name, dq_stat(method, scores))
        if isempty(scores)
            @warn "No data for $problem_name / $method (looked in $ddir/$pname)"
            continue
        end

        col = PALETTE[method]
        lbl = LABELS[method]
        agg = median_and_band(scores)
        isnothing(agg) && continue
        xs_med, med, lo, hi = agg

        maxx = INIT_DATA + target_iters(problem_name)
        keep = xs_med .<= maxx
        xs_med, med, lo, hi = xs_med[keep], med[keep], lo[keep], hi[keep]
        isempty(xs_med) && continue

        lines!(ax, xs_med, med; color=col, linewidth=2, label="$lbl")
    end

    return ax
end

# ─────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────

panels = [
    ("ABProblem",          "AB"),
    ("SimpleProblem",      "Simple"),
    ("BananaProblem",      "Banana"),
    ("BimodalProblem",     "Bimodal"),
    ("SIRProblem",         "SIR"),
    ("DuffingProblem",     "Duffing"),
    ("DiffusionProblem10", "Diffusion"),
]

ax_w, ax_h = 400, 300
ncols = 4
nrows = 2

mkpath(PLOT_DIR)

fig = Figure(; size = (ax_w * ncols + 60, ax_h * nrows + 40))

positions = [
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 2), (2, 3),
]

DQ_ACC = Dict{String,Vector{Any}}()
for (idx, (pname, title)) in enumerate(panels)
    r, c = positions[idx]
    @info "  [$r,$c] $pname ..."
    add_tv_panel!(fig[r, c], pname;
        title  = title,
        ylabel = (c == 1),
        dq_acc = DQ_ACC,
    )
end

ax_ref = let found = nothing
    for obj in fig.content
        if obj isa Axis && !isempty(obj.scene.plots)
            found = obj
            break
        end
    end
    found
end
if !isnothing(ax_ref)
    Legend(fig[2, 4], ax_ref; tellwidth=false, tellheight=false, labelsize=13, framevisible=true)
end

DQ_NOTES = String[]
for (p, s) in DQ_ACC
    append!(DQ_NOTES, dq_problem_notes(p, s))
end
sort!(DQ_NOTES)
dq_add_banner!(fig, DQ_NOTES)

# Manual, dated caveat — NOT auto-detected, so it needs its own banner (a different
# color from the red auto-checked one) or it'll be forgotten once the rerun below
# completes and nobody remembers to re-add it by hand.
dq_add_banner!(fig,
    ["SIRProblem/nongp plotted from 18/20 runs — runs 1-2 excluded, rerun in progress " *
     "since 2026-08-06 (jobs 11313592/11313593, ~cpulong 3d, expect ~2026-08-09). " *
     "Remove SIR_NONGP_EXCLUDE and re-plot once done."];
    row = -1, color = :darkorange, label = "KNOWN CAVEAT: ")

rowgap!(fig.layout, 15)
colgap!(fig.layout, 10)

save(joinpath(PLOT_DIR, "paper_surrogate_merged_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_surrogate_merged_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_surrogate_merged_tv.{png,pdf}"
