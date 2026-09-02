# Plot the final learned posteriors for all 24 opt-function problems.
# Produces two 4×6 grids: one for MaxVar run 1, one for EIV run 1.
# Same problem ordering as all other "all_opt_*" grids.

include("main.jl")

using CairoMakie
using JLD2

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

function add_learned_posterior_axis!(fig_pos, problem::AbstractProblem, run_name::String, run_idx::Int; grid_size=80)
    name = get_name(problem)
    prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")

    ax = Axis(fig_pos;
        title     = name,
        xlabel    = "x₁", ylabel = "x₂",
        titlesize = 13,
        xlabelsize = 11, ylabelsize = 11,
        xticklabelsize = 9, yticklabelsize = 9,
    )

    if !isfile(prob_file)
        text!(ax, 0.5, 0.5; text="no data", align=(:center, :center),
              space=:relative, fontsize=12, color=:red)
        return ax
    end

    bosip = load(prob_file)["problem"]
    log_post_mean = BOSIP.log_posterior_mean(bosip)

    bounds = domain(problem).bounds
    lb, ub = bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect

    # Build input matrix: x1 varies fastest → Z[i_x1, i_x2] after reshape
    XS = reduce(hcat, [x1, x2] for x2 in xs2 for x1 in xs1)
    log_vals = log_post_mean(XS)
    Z = reshape(exp.(log_vals), grid_size, grid_size)

    heatmap!(ax, xs1, xs2, Z; colormap=:matter)

    # Overlay queried points
    X_data = bosip.problem.data.X
    scatter!(ax, X_data[1, :], X_data[2, :];
        color=:white, markersize=3, strokewidth=0.5, strokecolor=:black)
    # Highlight last point in red
    scatter!(ax, [X_data[1, end]], [X_data[2, end]];
        color=:red, markersize=5)

    return ax
end

function plot_grid(run_name::String, run_idx::Int; grid_size=80)
    nrows, ncols = 4, 6
    cell_size = 220
    fig = Figure(; size = (cell_size * ncols, cell_size * nrows))

    for (i, prob) in enumerate(all_problems)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1
        name = get_name(prob)
        @info "  [$row,$col] $name ..."
        add_learned_posterior_axis!(fig[row, col], prob, run_name, run_idx; grid_size)
    end

    return fig
end

mkpath(plot_dir())

@info "Plotting learned posteriors — MaxVar run 1 ..."
fig_maxvar = plot_grid("maxvar", 1)
save(plot_dir() * "/all_learned_posteriors_maxvar1.png", fig_maxvar)
save(plot_dir() * "/all_learned_posteriors_maxvar1.pdf", fig_maxvar)
@info "Saved to $(plot_dir())/all_learned_posteriors_maxvar1.png"

@info "Plotting learned posteriors — EIV run 1 ..."
fig_eiv = plot_grid("eiv", 1)
save(plot_dir() * "/all_learned_posteriors_eiv1.png", fig_eiv)
save(plot_dir() * "/all_learned_posteriors_eiv1.pdf", fig_eiv)
@info "Saved to $(plot_dir())/all_learned_posteriors_eiv1.png"

@info "Done."
