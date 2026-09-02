include("plots.jl")

using CairoMakie
using JLD2

const INIT_DATA_HEX = 3
const GRID_SIZE = 200

problem = HexObsProblem(HimmelblauProblem())

function make_hex_bosip(problem, X, Y)
    data  = BOSS.ExperimentData(X, Y)
    model = GaussianProcess(;
        mean               = prior_mean(problem),
        kernel             = BOSS.Matern52Kernel(),
        lengthscale_priors = get_lengthscale_priors(problem),
        amplitude_priors   = get_amplitude_priors(problem),
        noise_std_priors   = get_noise_std_priors(problem),
    )
    return construct_bosip_problem(; problem, data, acquisition=LogMaxVar(), model)
end

function refit_hex!(bosip)
    fitter = OptimizationMAP(;
        algorithm=NEWUOA(), multistart=2, warm_start=true, parallel=false, rhoend=1e-4)
    BOSS.estimate_parameters!(bosip.problem, fitter; options=BossOptions(info=false))
end

function add_panel!(fig_pos, problem, run_name, run_idx, n_points; use_all=false, title="")
    ax = Axis(fig_pos;
        title = title,
        titlesize = 14,
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )
    prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")
    bosip  = load(prob_file)["problem"]
    X_full = bosip.problem.data.X
    Y_full = bosip.problem.data.Y
    N_avail = size(X_full, 2)
    N = use_all ? N_avail : min(n_points, N_avail)

    bosip.problem.data = BOSS.ExperimentData(X_full[:, 1:N], Y_full[:, 1:N])
    try
        refit_hex!(bosip)
        log_post_mean = BOSIP.log_posterior_mean(bosip)

        lb, ub = domain(problem).bounds
        xs1 = range(lb[1], ub[1]; length=GRID_SIZE) |> collect
        xs2 = range(lb[2], ub[2]; length=GRID_SIZE) |> collect
        XS  = reduce(hcat, [[x1, x2] for x2 in xs2 for x1 in xs1])
        Z   = reshape(exp.(log_post_mean(XS)), GRID_SIZE, GRID_SIZE)

        heatmap!(ax, xs1, xs2, Z; colormap=:matter)
        scatter!(ax, X_full[1, 1:N], X_full[2, 1:N];
                 color=:white, markersize=5, strokewidth=0.5, strokecolor=:black)
        scatter!(ax, [X_full[1, N]], [X_full[2, N]]; color=:red, markersize=8)
    catch e
        @warn "Panel error: $e"
        text!(ax, 0.5, 0.5; text="err", align=(:center,:center), space=:relative, fontsize=12, color=:orange)
    end
    return ax
end

function add_true_panel!(fig_pos, problem; title="true posterior")
    ax = Axis(fig_pos;
        title = title,
        titlesize = 14,
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )
    logpost_fn = true_logpost(problem)
    lb, ub = domain(problem).bounds
    xs1 = range(lb[1], ub[1]; length=GRID_SIZE) |> collect
    xs2 = range(lb[2], ub[2]; length=GRID_SIZE) |> collect
    log_ys = [logpost_fn([x1, x2]) for x1 in xs1, x2 in xs2]
    ys = exp.(log_ys .- maximum(log_ys))
    step1 = (ub[1] - lb[1]) / (GRID_SIZE - 1)
    step2 = (ub[2] - lb[2]) / (GRID_SIZE - 1)
    total  = sum(ys) - 0.5*(sum(ys[1,:]) + sum(ys[end,:]) + sum(ys[:,1]) + sum(ys[:,end]))
    total += 0.25*(ys[1,1] + ys[1,end] + ys[end,1] + ys[end,end])
    (total > 0) && (ys ./= step1 * step2 * total)
    heatmap!(ax, xs1, xs2, ys; colormap=:matter)
    x_true = true_params(problem)
    scatter!(ax, [x_true[1]], [x_true[2]]; color=:cyan, marker=:cross, markersize=10, strokewidth=1.5)
    return ax
end

mkpath(plot_dir())

snapshots = [5, 10, 20, 42]
n_points_list = INIT_DATA_HEX .+ snapshots

cw = 500
ncols = length(snapshots) + 1  # snapshots + true posterior
fig = Figure(; size = (cw * ncols, cw + 30))

for (i, (np, snap)) in enumerate(zip(n_points_list, snapshots))
    add_panel!(fig[1, i], problem, "eiv", 1, np; title="EIV iter $snap")
end
add_true_panel!(fig[1, ncols], problem)

colgap!(fig.layout, 4)

fname = plot_dir() * "/himmelblau_hex_eiv_zoom.png"
save(fname, fig)
@info "Saved to $fname"
