# Paper Figure 5: proxy variable comparison — loglike vs simulator output.
#
# All data from data-bosip-norm/ (normalized TV metric):
# standard        → data-bosip-norm/{Problem}/
# est             → data-bosip-norm/{Problem}/       (dashed orange, MAP estimator)
# loglike         → data-bosip-norm/Log{Problem}/    (solid blue)
# loglike-imiqr   → data-bosip-norm/Log{Problem}/    (dashed blue, Jarvenpaa state-of-the-art)
#
# Layout: 2×4 grid
#   Row 1: AB, Simple, Banana, Bimodal
#   Row 2: SIR, Duffing, Diffusion, (legend)
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

include("data_quality.jl")

const PLOT_DIR  = "plots"
const INIT_DATA = 3
const NORM_DIR  = "data-bosip-norm"

# Per-problem iteration target: 100 for the 2D problems, 200 for DuffingProblem
# (the one 3D problem in this group — harder, so it gets the higher HD-style
# target). Enforced explicitly here rather than relying on whichever run happens
# to be shortest on disk.
target_iters(problem_name::String) = problem_name == "DuffingProblem" ? 200 : 100

# ─────────────────────────────────────────────
# Data routing: (data_dir, problem_name) per method
# ─────────────────────────────────────────────

function log_problem_name(problem::String)
    problem == "DiffusionProblem10" && return "LogDiffusionProblem"
    return "Log" * problem
end

function method_source(method::String, problem::String)
    if method in ("standard", "est")
        return NORM_DIR, problem
    elseif method in ("loglike", "loglike-imiqr")
        return NORM_DIR, log_problem_name(problem)
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
# Palette & labels  (same slots as original plots.jl color_map)
# ─────────────────────────────────────────────

const METHODS = ["loglike", "loglike-imiqr", "standard", "est"]

## Canonical paper palette (2026-08-06, Wong/Okabe-Ito via Makie.wong_colors()):
## [1]=blue [2]=orange (reserved, niche custom-proxy plots) [3]=green [4]=pink
## [5]=light/sky blue [6]=vermillion [7]=yellow
const PALETTE = Dict(
    "standard"      => Makie.wong_colors()[2],   # orange (dark yellow) — MaxVar
    "est"           => Makie.wong_colors()[2],   # orange (dark yellow), dashed
    "loglike"       => Makie.wong_colors()[1],   # blue
    "loglike-imiqr" => Makie.wong_colors()[1],   # blue, dashed
)

const LINESTYLE = Dict(
    "standard"      => :solid,
    "est"           => :dash,
    "loglike"       => :solid,
    "loglike-imiqr" => :dash,
)

const LABELS = Dict(
    "standard"      => "GP - output - MaxVar - exp.",
    "est"           => "GP - output - MaxVar - MAP",
    "loglike"       => "GP - loglike - MaxVar - MAP",
    "loglike-imiqr" => "GP - loglike - IMIQR - MAP",
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
        scores = load_tv_scores(ddir, pname, method)
        isnothing(dq_acc) || dq_register!(dq_acc, problem_name, dq_stat(method, scores))
        if isempty(scores)
            @warn "No data for $problem_name / $method (looked in $ddir/$pname)"
            continue
        end

        col = PALETTE[method]
        lst = LINESTYLE[method]
        lbl = LABELS[method]
        agg = median_and_band(scores)
        isnothing(agg) && continue
        xs_med, med, lo, hi = agg

        maxx = INIT_DATA + target_iters(problem_name)
        keep = xs_med .<= maxx
        xs_med, med, lo, hi = xs_med[keep], med[keep], lo[keep], hi[keep]
        isempty(xs_med) && continue

        if lst == :dash
            lines!(ax, xs_med, med; color=col, linewidth=1.5, linestyle=:dash, label="$lbl")
        else
            lines!(ax, xs_med, med; color=col, linewidth=2, label="$lbl")
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

save(joinpath(PLOT_DIR, "paper_proxy_tv.png"), fig)
save(joinpath(PLOT_DIR, "paper_proxy_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/paper_proxy_tv.{png,pdf}"
