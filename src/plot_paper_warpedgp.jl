# Paper Figure: WarpedGP comparison — standard GP/MaxVar vs WarpedGP (YJ+Affine) / MaxVar.
#
# standard             → data-bosip-norm/{Problem}/  (20 runs)
# warpedgp-yja-maxvar  → data-warpedgp2/{Problem}/   (20 runs)
#
# Layout: 2×4 grid
#   Row 1: AB, Simple, Banana, Bimodal
#   Row 2: SIR, Duffing, Diffusion, (legend)
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR   = "plots"
const INIT_DATA  = 3
const NORM_DIR   = "data-bosip-norm"
const WARPEDGP_DIR = "data-warpedgp2"

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

function median_only(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] = s
    end
    valid = [i for i in 1:maxlen if all(!isnan, mat[i, :])]
    isempty(valid) && return nothing
    xs  = INIT_DATA .+ (valid .- 1)
    med = [median(mat[i, :]) for i in valid]
    return collect(xs), med
end

# ─────────────────────────────────────────────
# Palette & labels
# ─────────────────────────────────────────────

# Color: orange = standard GP; vermillion = WarpedGP
const PALETTE = Dict(
    "standard"            => Makie.wong_colors()[2],   # orange     — GP / MaxVar
    "warpedgp-yja-maxvar" => Makie.wong_colors()[6],   # vermillion — WarpedGP YJA / MaxVar
)

const LABELS = Dict(
    "standard"            => "GP - MaxVar",
    "warpedgp-yja-maxvar" => "WarpedGP YJA - MaxVar",
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

    for method in ["standard", "warpedgp-yja-maxvar"]
        data_dir = method == "standard" ? NORM_DIR : WARPEDGP_DIR
        max_runs = 20

        scores = load_tv_scores(data_dir, problem_name, method; max_runs)
        if isempty(scores)
            @warn "No data for $problem_name / $method"
            continue
        end

        col = PALETTE[method]
        lbl = LABELS[method]
        agg = median_only(scores)
        isnothing(agg) && continue
        xs, med = agg

        lines!(ax, xs, med; color=col, linewidth=2, label=lbl)
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
    Legend(fig[2, 4], ax_ref; tellwidth=false, tellheight=false, labelsize=13, framevisible=true,
        title="Surrogate")
end

rowgap!(fig.layout, 15)
colgap!(fig.layout, 10)

save(joinpath(PLOT_DIR, "paper_warpedgp_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_warpedgp_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_warpedgp_tv.{png,pdf}"
