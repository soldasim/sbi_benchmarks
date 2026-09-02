# True posteriors for the 12 5D cross-polytope optimization problems.
# Shows pairwise conditional slices (all other dims fixed at true_params) in a 2×6 × 5×5 grid.
# Lower triangle: 2D heatmap varying (col, row) dims. Diagonal: 1D slice. Upper triangle: empty.

include("plots.jl")

mkpath(plot_dir())

all_problems = CrossPolytopeObsProblem.([
    RosenbrockProblem(; x_dim=5),
    StyblinskiTangProblem(; x_dim=5),
    MichalewiczProblem(; x_dim=5),
    AckleyProblem(; x_dim=5),
    AlpineProblem(; x_dim=5),
    ExpandedSchafferF6Problem(; x_dim=5),
    ExpandedZakharovProblem(; x_dim=5),
    GriewankProblem(; x_dim=5),
    RastriginProblem(; x_dim=5),
    SalomonProblem(; x_dim=5),
    SchwefelProblem(; x_dim=5),
    SphereProblem(; x_dim=5),
])

@assert length(all_problems) == 12

nrows_outer, ncols_outer = 2, 6
ndim = 5
grid_size = 60
panel_px = 100
title_px = 16

fig = Figure(size = (panel_px * ndim * ncols_outer, (panel_px * ndim + title_px) * nrows_outer))

@info "Building 2×6 grid of 5×5 conditional slice plots for 5D cross-polytope problems ..."

for (k, prob) in enumerate(all_problems)
    row_outer = div(k - 1, ncols_outer) + 1
    col_outer = mod(k - 1, ncols_outer) + 1
    name = get_name(prob)
    @info "  ($row_outer,$col_outer) $name"

    logpost_fn = true_logpost(prob)
    lb, ub = domain(prob).bounds
    x_true = true_params(prob)

    xs = [range(lb[i], ub[i]; length=grid_size) |> collect for i in 1:ndim]

    gl = GridLayout()
    fig[row_outer, col_outer] = gl

    Label(gl[0, 1:ndim]; text=name, fontsize=8, tellwidth=false, halign=:center)

    for row_inner in 1:ndim, col_inner in 1:ndim
        col_inner > row_inner && continue  # upper triangle — leave empty

        ax = Axis(gl[row_inner, col_inner];
            xticklabelsvisible = false,
            yticklabelsvisible = false,
            xticksvisible = false,
            yticksvisible = false,
        )

        if row_inner == col_inner
            dim = row_inner
            log_ys = map(xs[dim]) do xi
                x_v = copy(x_true)
                x_v[dim] = xi
                logpost_fn(x_v)
            end
            ys = exp.(log_ys .- maximum(log_ys))
            lines!(ax, xs[dim], ys; color=:black, linewidth=0.8)
            vlines!(ax, [x_true[dim]]; color=:cyan, linewidth=1.0)
        else
            dim_x = col_inner
            dim_y = row_inner
            log_zs = [begin
                x_v = copy(x_true)
                x_v[dim_x] = xi
                x_v[dim_y] = yi
                logpost_fn(x_v)
            end for xi in xs[dim_x], yi in xs[dim_y]]
            zs = exp.(log_zs .- maximum(log_zs))
            heatmap!(ax, xs[dim_x], xs[dim_y], zs; colormap=:matter)
            scatter!(ax, [x_true[dim_x]], [x_true[dim_y]];
                     color=:cyan, marker=:cross, markersize=6, strokewidth=1.0)
        end
    end

    colgap!(gl, 2)
    rowgap!(gl, 2)
end

colgap!(fig.layout, 6)
rowgap!(fig.layout, 6)

@info "Saving ..."
save(plot_dir() * "/cross5d_true_posteriors.png", fig)
save(plot_dir() * "/cross5d_true_posteriors.pdf", fig)
@info "Done. Saved to $(plot_dir())/cross5d_true_posteriors.png"
