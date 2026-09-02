# TV metric convergence comparison on the 7 original benchmarks (ProxySIR instead of SIR)
# with the normalized TV metric recomputation.
#
# All data loads from data-bosip-norm/.
# Missing files are silently skipped — run whenever partial data is available.
#
# Layout: 2×4 grid
#   Row 1: ABProblem, SimpleProblem, BananaProblem, BimodalProblem
#   Row 2: ProxySIRProblem, DuffingProblem, DiffusionProblem10, (legend)
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

include("data_quality.jl")

const PLOT_DIR   = "plots"
const INIT_DATA  = 3
const DATA_DIR   = "data-bosip-norm"

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
    valid = [i for i in 1:maxlen if any(!isnan, mat[i, :])]
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

const METHODS = ["standard", "eiv", "immd", "nongp", "eiig"]

## Canonical paper palette (2026-08-06, Wong/Okabe-Ito via Makie.wong_colors()):
## [1]=blue [2]=orange (reserved, niche custom-proxy plots) [3]=green [4]=pink
## [5]=light/sky blue [6]=vermillion (warpedgp, reserved even when not in this
## figure) [7]=yellow
const PALETTE = Dict(
    "standard" => Makie.wong_colors()[2],   # orange (dark yellow) — MaxVar
    "eiv"      => Makie.wong_colors()[3],   # green  — EIV
    "immd"     => Makie.wong_colors()[4],   # pink   — new IMMD (same as eiig)
    "nongp"    => Makie.wong_colors()[5],   # light/sky blue — NonGP (fixed: was vermillion, now consistent with the other surrogate plots)
    "eiig"     => (:gray, 0.5),             # gray, dashed — old IMMD (reference)
)

const LABELS = Dict(
    "standard" => "MaxVar",
    "eiv"      => "EIV",
    "immd"     => "IMMD (new)",
    "nongp"    => "NonGP",
    "eiig"     => "IMMD (old)",
)

method_data_dir(method::String) = DATA_DIR

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
        ddir   = method_data_dir(method)
        scores = load_tv_scores(ddir, problem_name, method)
        (isnothing(dq_acc) || method == "eiig") || dq_register!(dq_acc, problem_name, dq_stat(method, scores))
        if isempty(scores)
            @warn "No data for $problem_name / $method (looked in $ddir)"
            continue
        end

        col = PALETTE[method]
        lbl = LABELS[method]
        n   = length(scores)
        agg = median_and_band(scores)
        isnothing(agg) && continue
        xs_med, med, lo, hi = agg

        if method == "eiig"
            lines!(ax, xs_med, med; color=col, linewidth=1.5, linestyle=:dash, label="$lbl (n=$n)")
        else
            maxlen = maximum(length.(scores))
            xs_full = collect(INIT_DATA:(INIT_DATA + maxlen - 1))
            for s in scores
                lines!(ax, xs_full[eachindex(s)], s; color=(col isa Tuple ? col[1] : col, 0.15), linewidth=0.6)
            end
            band!(ax, xs_med, lo, hi; color=(col isa Tuple ? col[1] : col, 0.2))
            lines!(ax, xs_med, med; color=col, linewidth=2, label="$lbl (n=$n)")
        end
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
    ("ProxySIRProblem",    "Proxy SIR"),
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

local ax_last = nothing
DQ_ACC = Dict{String,Vector{Any}}()
for (idx, (pname, title)) in enumerate(panels)
    r, c = positions[idx]
    @info "  [$r,$c] $pname ..."
    ax = add_tv_panel!(fig[r, c], pname;
        title  = title,
        ylabel = (c == 1),
        dq_acc = DQ_ACC,
    )
    ax_last = ax
end

DQ_NOTES = String[]
for (p, s) in DQ_ACC
    append!(DQ_NOTES, dq_problem_notes(p, s))
end
sort!(DQ_NOTES)
dq_add_banner!(fig, DQ_NOTES)

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

rowgap!(fig.layout, 15)
colgap!(fig.layout, 10)

save(joinpath(PLOT_DIR, "bosip_norm_tv.png"), fig)
save(joinpath(PLOT_DIR, "bosip_norm_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/bosip_norm_tv.{png,pdf}"
