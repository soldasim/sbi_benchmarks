# Plot the true posteriors for all 24 opt-function benchmark problems in a 4×6 grid.
# Same problem ordering as plot_all_opt_results.jl.

include("main.jl")

using CairoMakie
using Random

Random.seed!(42)

all_problems = [
    # Original 3
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
    # New d-dimensional at 2D
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
    # New 2D-only
    BealeProblem(),
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProblem(),
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
]

@assert length(all_problems) == 24

function add_posterior_axis!(fig_pos, problem::AbstractProblem; grid_size=80)
    bounds     = domain(problem).bounds
    logpost_fn = true_logpost(problem)
    lb, ub     = bounds

    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect

    log_ys = [logpost_fn([x1, x2]) for x1 in xs1, x2 in xs2]
    ys = exp.(log_ys .- maximum(log_ys))
    step1 = (ub[1] - lb[1]) / (grid_size - 1)
    step2 = (ub[2] - lb[2]) / (grid_size - 1)
    total = sum(ys) - 0.5*(sum(ys[1,:]) + sum(ys[end,:]) + sum(ys[:,1]) + sum(ys[:,end]))
    total += 0.25*(ys[1,1] + ys[1,end] + ys[end,1] + ys[end,end])
    (total > 0) && (ys ./= step1 * step2 * total)

    ax = Axis(fig_pos;
        title = get_name(problem),
        xlabel = "x₁", ylabel = "x₂",
        titlesize = 13,
        xlabelsize = 11, ylabelsize = 11,
        xticklabelsize = 9, yticklabelsize = 9,
    )
    heatmap!(ax, xs1, xs2, ys; colormap=:matter)
    return ax
end

mkpath(plot_dir())

@info "Plotting 4×6 grid of all 24 true posteriors ..."

nrows, ncols = 4, 6
cell_size = 220
fig = Figure(; size = (cell_size * ncols, cell_size * nrows))

for (i, prob) in enumerate(all_problems)
    row = div(i - 1, ncols) + 1
    col = mod(i - 1, ncols) + 1
    name = get_name(prob)
    @info "  [$row,$col] $name ..."
    add_posterior_axis!(fig[row, col], prob; grid_size=80)
end

save(plot_dir() * "/all_opt_posteriors_grid.png", fig)
save(plot_dir() * "/all_opt_posteriors_grid.pdf", fig)
@info "Done. Saved to $(plot_dir())/all_opt_posteriors_grid.png"
