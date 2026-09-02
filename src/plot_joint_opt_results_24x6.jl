# Joint 24×6 grid: rows = problems, cols = [base TV, base posterior, sharp TV, sharp posterior, hex TV, hex posterior].
# Shows TV-metric convergence (MaxVar + EIV, all runs + median) and true posteriors across all three experiment variants.

include("plots.jl")

using CairoMakie

mkpath(plot_dir())

# SharpProblem delegates all AbstractProblem methods to its base except true_params.
true_params(p::SharpProblem) = true_params(p.base)

const _GROUPS    = ["maxvar", "eiv"]
const _GRID_SIZE = 80

# --- Problem lists ---

const BASE_PROBLEMS = [
    # Original 3
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
    # d-dimensional at 2D
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
    # 2D-only (proxy variants for Beale and GoldsteinPrice)
    BealeProxyProblem(),
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProxyProblem(),
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
]

@assert length(BASE_PROBLEMS) == 24

const SHARP_PROBLEMS = SharpProblem.(BASE_PROBLEMS)
const HEX_PROBLEMS   = HexObsProblem.(BASE_PROBLEMS)

# --- Posterior panel helper ---

function add_posterior_axis!(fig_pos, problem::AbstractProblem; grid_size=_GRID_SIZE)
    logpost_fn = true_logpost(problem)
    lb, ub = domain(problem).bounds

    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect

    log_ys = [logpost_fn([x1, x2]) for x1 in xs1, x2 in xs2]
    ys = exp.(log_ys .- maximum(log_ys))
    step1 = (ub[1] - lb[1]) / (grid_size - 1)
    step2 = (ub[2] - lb[2]) / (grid_size - 1)
    total  = sum(ys) - 0.5*(sum(ys[1,:]) + sum(ys[end,:]) + sum(ys[:,1]) + sum(ys[:,end]))
    total += 0.25*(ys[1,1] + ys[1,end] + ys[end,1] + ys[end,end])
    (total > 0) && (ys ./= step1 * step2 * total)

    ax = Axis(fig_pos;
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )
    heatmap!(ax, xs1, xs2, ys; colormap=:matter)

    x_true = true_params(problem)
    scatter!(ax, [x_true[1]], [x_true[2]]; color=:cyan, marker=:cross, markersize=8, strokewidth=1.2)

    return ax
end

# --- Figure layout ---
# Row 1: column headers
# Rows 2..25: one per problem (24 rows)
# Columns 1,3,5: TV metric (width = ax_w), columns 2,4,6: posterior (square, width = ax_h)

_, ax_h = axis_size()       # (400, 300) — use ax_h for row height only
ax_w   = 400               # TV column content width (Fixed)
post_w = ax_h              # 300 — square posterior cells

header_h = 50

set_theme_fonts!(; base_fontsize=14)

# Size is set to a rough initial value; resize_to_layout! after populating will
# expand the figure to fit axis protrusions (y-labels, tick labels) exactly.
fig = Figure(; size=(3 * ax_w + 3 * post_w, header_h + 24 * ax_h))

# Column headers must be added first so the layout knows all 6 columns exist,
# after which colsize!/rowsize! can be called.
col_headers = [
    "Base — TV metric",
    "Base — posterior",
    "Sharp — TV metric",
    "Sharp — posterior",
    "Hex — TV metric",
    "Hex — posterior",
]
for (c, txt) in enumerate(col_headers)
    Label(fig[1, c]; text=txt, fontsize=13, font=:bold, tellwidth=false, halign=:center)
end

for c in (1, 3, 5); colsize!(fig.layout, c, Fixed(ax_w));   end
for c in (2, 4, 6); colsize!(fig.layout, c, Fixed(post_w)); end
rowsize!(fig.layout, 1, Fixed(header_h))

# Problem rows
for (i, (bp, sp, hp)) in enumerate(zip(BASE_PROBLEMS, SHARP_PROBLEMS, HEX_PROBLEMS))
    row = i + 1
    @info "Row $i/24: $(get_name(bp)) ..."

    plot_result_axis!(fig[row, 1], [bp];
        legend=false, metric=:tv, plotted_groups=_GROUPS, plot_individual_runs=true)
    add_posterior_axis!(fig[row, 2], bp)

    plot_result_axis!(fig[row, 3], [sp];
        legend=false, metric=:tv, plotted_groups=_GROUPS, plot_individual_runs=true)
    add_posterior_axis!(fig[row, 4], sp)

    plot_result_axis!(fig[row, 5], [hp];
        legend=(i == 1), metric=:tv, plotted_groups=_GROUPS, plot_individual_runs=true)
    add_posterior_axis!(fig[row, 6], hp)
end

colgap!(fig.layout, 4)
rowgap!(fig.layout, 4)
resize_to_layout!(fig)

@info "Saving joint_opt_results_24x6 ..."
save(plot_dir() * "/joint_opt_results_24x6.png", fig)
save(plot_dir() * "/joint_opt_results_24x6.pdf", fig)
@info "Done. Saved to $(plot_dir())/joint_opt_results_24x6.png"
