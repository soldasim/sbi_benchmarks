# TV metric convergence comparison: MaxVar vs EIV vs new IMMD (immd run_name).
# Old IMMD (eiig) is included as a dashed reference line.
#
# All data lives in data-bosip-norm/<Problem>/.
#
# Layout: 2×4 grid
#   Row 1: ABProblem, SimpleProblem, BananaProblem, BimodalProblem
#   Row 2: ProxySIRProblem, DuffingProblem, DiffusionProblem10, (legend)
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR    = "plots"
const INIT_DATA   = 3
const DATA_DIR    = "data-bosip-norm"

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
    xs  = INIT_DATA:(INIT_DATA + maxlen - 1)
    med = [median(filter(!isnan, mat[i, :])) for i in 1:maxlen]
    lo  = [quantile(filter(!isnan, mat[i, :]), 0.25) for i in 1:maxlen]
    hi  = [quantile(filter(!isnan, mat[i, :]), 0.75) for i in 1:maxlen]
    return collect(xs), med, lo, hi
end

# ─────────────────────────────────────────────
# Palette & labels
# ─────────────────────────────────────────────

const METHODS = ["standard", "eiv", "immd", "eiig"]

const PALETTE = Dict(
    "standard" => Makie.wong_colors()[2],   # orange — MaxVar
    "eiv"      => Makie.wong_colors()[3],   # green  — EIV
    "immd"     => Makie.wong_colors()[4],   # pink   — new IMMD (same as eiig)
    "eiig"     => (:gray, 0.5),             # gray, dashed — old IMMD (reference)
)

const LABELS = Dict(
    "standard" => "MaxVar",
    "eiv"      => "EIV",
    "immd"     => "IMMD (new)",
    "eiig"     => "IMMD (old)",
)

# Map method → data directory
method_data_dir(method::String) = DATA_DIR

# ─────────────────────────────────────────────
# Per-panel plotting
# ─────────────────────────────────────────────

function add_tv_panel!(figpos, problem_name::String; title="", legend=false, ylabel=true, show_eiig=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        xscale = log10,
        yscale = log10,
    )

    plotted_methods = show_eiig ? METHODS : filter(≠("eiig"), METHODS)

    for method in plotted_methods
        ddir  = method_data_dir(method)
        scores = load_tv_scores(ddir, problem_name, method)
        if isempty(scores)
            @warn "No data for $problem_name / $method (looked in $ddir)"
            continue
        end

        col  = PALETTE[method]
        lbl  = LABELS[method]
        n    = length(scores)
        agg  = median_and_band(scores)
        isnothing(agg) && continue
        xs_med, med, lo, hi = agg

        if method == "eiig"
            # Old IMMD as dashed gray reference (no band, no individual runs)
            lines!(ax, xs_med, med; color=col, linewidth=1.5, linestyle=:dash, label="$lbl (n=$n)")
        else
            # Individual runs (thin, semi-transparent)
            maxlen = maximum(length.(scores))
            xs_full = collect(INIT_DATA:(INIT_DATA + maxlen - 1))
            for s in scores
                lines!(ax, xs_full[eachindex(s)], s; color=(col isa Tuple ? col[1] : col, 0.15), linewidth=0.6)
            end
            # IQR band
            band!(ax, xs_med, lo, hi; color=(col isa Tuple ? col[1] : col, 0.2))
            # Median
            lines!(ax, xs_med, med; color=col, linewidth=2, label="$lbl (n=$n)")
        end
    end

    legend && axislegend(ax; position=:lb, labelsize=11, merge=true)
    return ax
end

# ─────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────

# (problem_name, display_title)
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

# Grid positions (row-major, last cell of row 2 = legend)
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
        legend = false,
        show_eiig = true,
    )
end

# Legend in the last cell (row 2, col 4)
# Pull it from the first axis that has entries
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

save(joinpath(PLOT_DIR, "immd_comparison_tv_proxysir.png"), fig)
save(joinpath(PLOT_DIR, "immd_comparison_tv_proxysir.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/immd_comparison_tv_proxysir.{png,pdf}"
