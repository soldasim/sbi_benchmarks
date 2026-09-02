# True posterior corner plot for DuffingProblem5 (5-parameter Duffing oscillator).
# Lower-triangle + diagonal layout: pairwise LHC marginals from precomputed grid.
#
# Phase 1 — precompute (run once on the cluster, ~minutes with 4 threads):
#   include("src/plot_marginals.jl")
#   precompute_bip_grid(DuffingProblem5())
#
# Phase 2 — plot (cheap, re-run freely):
#   include("src/plot_duffing5_true_posteriors.jl")

include("plots.jl")
include("plot_marginals.jl")

mkpath(plot_dir())

prob   = DuffingProblem5()
name   = get_name(prob)
labs   = _param_labels(prob)
ndim   = x_dim(prob)   # 5
x_true = true_params(prob)

data   = _load_grid(prob)
xs     = data["xs_per_dim"]
pdims  = data["pair_dims"]
pmarg  = data["pair_marginals"]
m1d    = data["marg1d"]

panel_px = 120
gap_px   = 4

fig = Figure(size = (panel_px * ndim + 40, panel_px * ndim + 40))

@info "Plotting $name corner plot ..."

for row in 1:ndim, col in 1:ndim
    col > row && continue   # upper triangle — leave empty

    local ax = Axis(fig[row, col];
        xticklabelsvisible = (row == ndim),
        yticklabelsvisible = (col == 1 && row != col),
        xticksvisible      = (row == ndim),
        yticksvisible      = (col == 1 && row != col),
        xticklabelsize     = 9,
        yticklabelsize     = 9,
        xlabel             = row == ndim ? labs[col] : "",
        ylabel             = col == 1 && row != col ? labs[row] : "",
        xlabelsize         = 11,
        ylabelsize         = 11,
    )

    if row == col
        # Diagonal: 1-D marginal of parameter `row`
        lv = m1d[row]
        ys = exp.(lv .- maximum(lv))
        lines!(ax, xs[row], ys; color=:black, linewidth=1.2)
        vlines!(ax, [x_true[row]]; color=:cyan, linewidth=1.2)
    else
        # Lower triangle: 2-D marginal of (col, row)
        pi = findfirst(pd -> pd[1] == col && pd[2] == row, pdims)
        lv = pmarg[pi]
        zs = exp.(lv .- maximum(lv))
        heatmap!(ax, xs[col], xs[row], zs; colormap=:matter)
        scatter!(ax, [x_true[col]], [x_true[row]];
                 color=:cyan, marker=:cross, markersize=8, strokewidth=1.2)
    end
end

colgap!(fig.layout, gap_px)
rowgap!(fig.layout, gap_px)

@info "Saving ..."
save(plot_dir() * "/duffing5_true_posteriors.png", fig)
save(plot_dir() * "/duffing5_true_posteriors.pdf", fig)
@info "Done. Saved to $(plot_dir())/duffing5_true_posteriors.{png,pdf}"
