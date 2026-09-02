# True posteriors for all 24 2D cross-polytope opt-function problems (4×6 grid).
# No experiment data required — evaluates true_logpost analytically on a grid.

include("plots.jl")

mkpath(plot_dir())

all_cross_problems = CrossPolytopeObsProblem.([
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
    # 2D-only
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
])

@assert length(all_cross_problems) == 24

nrows, ncols = 4, 6
grid_size = 80
cw, ch = 220, 220

@info "Building 4×6 true posterior grid for 2D cross-polytope problems ..."

fig = Figure(; size = (cw * ncols, ch * nrows))

for (i, prob) in enumerate(all_cross_problems)
    row = div(i - 1, ncols) + 1
    col = mod(i - 1, ncols) + 1
    name = get_name(prob)
    @info "  ($row,$col) $name"

    ax = Axis(fig[row, col];
        title = name,
        titlesize = 9,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        xticksvisible = false,
        yticksvisible = false,
    )

    logpost_fn = true_logpost(prob)
    lb, ub = domain(prob).bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect
    log_ys = [logpost_fn([x1, x2]) for x1 in xs1, x2 in xs2]
    ys = exp.(log_ys .- maximum(log_ys))

    heatmap!(ax, xs1, xs2, ys; colormap=:matter)

    x_true = true_params(prob)
    scatter!(ax, [x_true[1]], [x_true[2]];
             color=:cyan, marker=:cross, markersize=10, strokewidth=1.5)
end

colgap!(fig.layout, 4)
rowgap!(fig.layout, 4)

save(plot_dir() * "/cross_true_posteriors.png", fig)
save(plot_dir() * "/cross_true_posteriors.pdf", fig)
@info "Done. Saved to $(plot_dir())/cross_true_posteriors.png"
