# TV metric comparison for the new 5D physical BIP problems:
# DuffingProblem5 and DiffusionProblem5D.
#
# Methods plotted: standard (MaxVar), maxvar, eiv, immd.
# Partial data is handled gracefully — only iterations where ALL 5 runs
# have data are included in the median line.
#
# Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.

using JLD2
using CairoMakie
using Statistics

const PLOT_DIR_5D = "plots"
const INIT_DATA_5D = 3
const DATA_DIR_5D  = "data-bosip-norm"

function load_tv_scores_5d(problem::String, method::String; max_runs::Int=5)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        fpath = joinpath(DATA_DIR_5D, problem, "$(method)_$(i)_TVmetric.jld2")
        isfile(fpath) || continue
        s = load(fpath, "score")
        isnothing(s) || push!(scores, s)
    end
    return scores
end

function median_and_band_5d(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] = s
    end
    valid = [i for i in 1:maxlen if all(!isnan, mat[i, :])]
    isempty(valid) && return nothing
    xs  = INIT_DATA_5D .+ (valid .- 1)
    med = [median(filter(!isnan, mat[i, :])) for i in valid]
    lo  = [quantile(filter(!isnan, mat[i, :]), 0.25) for i in valid]
    hi  = [quantile(filter(!isnan, mat[i, :]), 0.75) for i in valid]
    return collect(xs), med, lo, hi
end

const METHODS_5D = ["standard", "maxvar", "eiv", "immd"]

const PALETTE_5D = Dict(
    "standard" => Makie.wong_colors()[2],   # orange  — MaxVar (standard)
    "maxvar"   => Makie.wong_colors()[1],   # blue    — MaxVar (maxvar variant)
    "eiv"      => Makie.wong_colors()[3],   # green   — EIV
    "immd"     => Makie.wong_colors()[4],   # pink    — IMMD
)

const LABELS_5D = Dict(
    "standard" => "MaxVar (standard)",
    "maxvar"   => "MaxVar",
    "eiv"      => "EIV",
    "immd"     => "IMMD",
)

function add_tv_panel_5d!(figpos, problem_name::String; title="", ylabel=true)
    ax = Axis(figpos;
        xlabel = "simulations",
        ylabel = ylabel ? "TV distance" : "",
        title  = title,
        xscale = log10,
        yscale = log10,
    )

    for method in METHODS_5D
        scores = load_tv_scores_5d(problem_name, method)
        isempty(scores) && continue

        agg = median_and_band_5d(scores)
        isnothing(agg) && continue
        xs_med, med, lo, hi = agg

        col = PALETTE_5D[method]
        lbl = LABELS_5D[method]
        n   = length(scores)

        lines!(ax, xs_med, med; color=col, linewidth=2, label="$lbl (n=$n)")
        band!(ax, xs_med, lo, hi; color=(col, 0.2))
    end

    return ax
end

panels_5d = [
    ("DuffingProblem5",   "Duffing (5D)"),
    ("DiffusionProblem5D", "Diffusion (5D)"),
]

ax_w, ax_h = 450, 340
mkpath(PLOT_DIR_5D)

fig = Figure(; size = (ax_w * 2 + 240, ax_h + 60))

for (idx, (pname, title)) in enumerate(panels_5d)
    @info "  Plotting $pname ..."
    ax = add_tv_panel_5d!(fig[1, idx], pname;
        title  = title,
        ylabel = (idx == 1),
    )
    if idx == 1
        axislegend(ax; position=:rt, labelsize=12, framevisible=true)
    end
end

colgap!(fig.layout, 15)

save(joinpath(PLOT_DIR_5D, "5d_bip_tv.png"), fig)
save(joinpath(PLOT_DIR_5D, "5d_bip_tv.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR_5D)/5d_bip_tv.{png,pdf}"
