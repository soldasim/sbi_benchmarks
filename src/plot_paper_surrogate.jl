# Paper Figure 8: surrogate model comparison — stationary GP vs nonstationary GP.
#
# standard → data-bosip-norm/{Problem}/   (GP, MaxVar)
# nongp    → data-bosip-norm/{Problem}/   (nonGP, MaxVar)
#
# Layout: 2×4 grid
#   Row 1: AB, Simple, Banana, Bimodal
#   Row 2: SIR, Duffing, Diffusion, (legend)
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR  = "plots"
const INIT_DATA = 3
const DATA_DIR  = "data-bosip-norm"

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
# Palette & labels  (same slots as original plots.jl color_map)
# ─────────────────────────────────────────────

const METHODS = ["standard", "nongp"]

const PALETTE = Dict(
    "standard" => Makie.wong_colors()[2],   # orange — GP / MaxVar
    "nongp"    => Makie.wong_colors()[5],   # sky blue — nonGP / MaxVar
)

const LABELS = Dict(
    "standard" => "GP - output - MaxVar",
    "nongp"    => "nonGP - output - MaxVar",
)

# ─────────────────────────────────────────────
# Per-panel plotting
# ─────────────────────────────────────────────

function add_tv_panel!(figpos, problem_name::String; title="", ylabel=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        xscale = log10,
        yscale = log10,
    )

    for method in METHODS
        scores = load_tv_scores(DATA_DIR, problem_name, method)
        if isempty(scores)
            @warn "No data for $problem_name / $method"
            continue
        end

        col = PALETTE[method]
        lbl = LABELS[method]
        n   = length(scores)
        agg = median_and_band(scores)
        isnothing(agg) && continue
        xs_med, med, lo, hi = agg

        lines!(ax, xs_med, med; color=col, linewidth=2, label="$lbl (n=$n)")
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

for (idx, (pname, title)) in enumerate(panels)
    r, c = positions[idx]
    @info "  [$r,$c] $pname ..."
    add_tv_panel!(fig[r, c], pname;
        title  = title,
        ylabel = (c == 1),
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

rowgap!(fig.layout, 15)
colgap!(fig.layout, 10)

save(joinpath(PLOT_DIR, "paper_surrogate_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_surrogate_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_surrogate_tv.{png,pdf}"
