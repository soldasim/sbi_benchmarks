# Plot learned posteriors for MaxVar run 1 at 50 and 100 iterations.
# Uses final fitted hyperparameters but truncates data to first N points.

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

function add_snapshot_axis!(fig_pos, problem::AbstractProblem, run_name::String, run_idx::Int, n_points::Int; grid_size=80)
    name = get_name(problem)
    prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")

    ax = Axis(fig_pos;
        title      = name,
        xlabel     = "x₁", ylabel = "x₂",
        titlesize  = 13,
        xlabelsize = 11, ylabelsize = 11,
        xticklabelsize = 9, yticklabelsize = 9,
    )

    if !isfile(prob_file)
        text!(ax, 0.5, 0.5; text="no data", align=(:center, :center),
              space=:relative, fontsize=12, color=:red)
        return ax
    end

    bosip  = load(prob_file)["problem"]
    X_full = bosip.problem.data.X
    Y_full = bosip.problem.data.Y
    N      = min(n_points, size(X_full, 2))

    # Truncate to first N points (use final fitted hyperparams)
    bosip.problem.data = BOSS.ExperimentData(X_full[:, 1:N], Y_full[:, 1:N])

    log_post_mean = BOSIP.log_posterior_mean(bosip)

    bounds = domain(problem).bounds
    lb, ub = bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect

    XS       = reduce(hcat, [x1, x2] for x2 in xs2 for x1 in xs1)
    log_vals = log_post_mean(XS)
    Z        = reshape(exp.(log_vals), grid_size, grid_size)

    heatmap!(ax, xs1, xs2, Z; colormap=:matter)
    scatter!(ax, X_full[1, 1:N], X_full[2, 1:N];
        color=:white, markersize=3, strokewidth=0.5, strokecolor=:black)
    scatter!(ax, [X_full[1, N]], [X_full[2, N]]; color=:red, markersize=5)

    return ax
end

function plot_snapshot_grid(run_name::String, run_idx::Int, n_points::Int; grid_size=80)
    nrows, ncols = 4, 6
    fig = Figure(; size = (220 * ncols, 220 * nrows))
    for (i, prob) in enumerate(all_problems)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1
        @info "  [$(row),$(col)] $(get_name(prob)) ..."
        add_snapshot_axis!(fig[row, col], prob, run_name, run_idx, n_points; grid_size)
    end
    return fig
end

mkpath(plot_dir())

for iter_snapshot in [50, 100]
    n_points = 3 + iter_snapshot  # 3 initial data points + iter BO steps
    @info "Plotting MaxVar run 1 — iter $iter_snapshot (n=$n_points) ..."
    fig = plot_snapshot_grid("maxvar", 1, n_points)
    fname = plot_dir() * "/learned_posteriors_maxvar1_iter$(iter_snapshot).png"
    save(fname, fig)
    save(plot_dir() * "/learned_posteriors_maxvar1_iter$(iter_snapshot).pdf", fig)
    @info "Saved to $fname"
end

@info "Done."
