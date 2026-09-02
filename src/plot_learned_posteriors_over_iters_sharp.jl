# Plot GP-learned posteriors at 25, 50, 100, 200 iterations + true posteriors
# for all 24 sharp opt-function problems (4×6 grid per snapshot).
# Generates one grid per (run_name, n_points) combination.

include("main.jl")

using CairoMakie
using JLD2

const SHARP_RUN_IDX = 1   # which run to visualise

all_problems = SharpProblem.([
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
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

@assert length(all_problems) == 24

function add_true_posterior_axis!(fig_pos, problem::AbstractProblem; grid_size=80)
    bounds     = domain(problem).bounds
    logpost_fn = true_logpost(problem)
    lb, ub     = bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect
    log_ys = [logpost_fn([x1, x2]) for x1 in xs1, x2 in xs2]
    ys = exp.(log_ys .- maximum(log_ys))
    ax = Axis(fig_pos;
        title = get_name(problem) * " (true)",
        xlabel = "x₁", ylabel = "x₂",
        titlesize = 11, xlabelsize = 9, ylabelsize = 9,
        xticklabelsize = 8, yticklabelsize = 8,
    )
    heatmap!(ax, xs1, xs2, ys; colormap=:matter)
    return ax
end

function add_snapshot_axis!(fig_pos, problem::AbstractProblem, run_name::String, run_idx::Int, n_points::Int; grid_size=80)
    name = get_name(problem)
    prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")

    ax = Axis(fig_pos;
        title = name,
        xlabel = "x₁", ylabel = "x₂",
        titlesize = 11, xlabelsize = 9, ylabelsize = 9,
        xticklabelsize = 8, yticklabelsize = 8,
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

function plot_true_posterior_grid(; grid_size=80)
    nrows, ncols = 4, 6
    fig = Figure(; size = (220 * ncols, 220 * nrows))
    for (i, prob) in enumerate(all_problems)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1
        @info "  [$(row),$(col)] $(get_name(prob)) ..."
        add_true_posterior_axis!(fig[row, col], prob; grid_size)
    end
    return fig
end

mkpath(plot_dir())

# True posteriors
@info "Plotting true posteriors (sharp) ..."
fig = plot_true_posterior_grid()
save(plot_dir() * "/sharp_posteriors_true.png", fig)
@info "Saved sharp_posteriors_true.png"

# Iteration snapshots for MaxVar and EIV
for run_name in ["maxvar", "eiv"]
    for iter_snapshot in [25, 50, 100, 200]
        n_points = 3 + iter_snapshot
        @info "Plotting $run_name run $SHARP_RUN_IDX — iter $iter_snapshot (n=$n_points) ..."
        fig = plot_snapshot_grid(run_name, SHARP_RUN_IDX, n_points)
        fname = plot_dir() * "/sharp_$(run_name)$(SHARP_RUN_IDX)_iter$(iter_snapshot).png"
        save(fname, fig)
        @info "Saved $fname"
    end
end

@info "All progression plots done."
