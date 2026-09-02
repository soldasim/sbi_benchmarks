# TV-metric convergence + learned-posterior progression for the 3 hex opt-function problems.
# Run from the repo root after experiments have finished.

include("plots.jl")

using CairoMakie
using JLD2

mkpath(plot_dir())

const INIT_DATA_HEX = 3
const HEX_PROBLEMS = HexObsProblem.([
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
])
const HEX_GROUPS = ["maxvar", "eiv"]

# ─── TV-metric convergence (3 stacked panels) ─────────────────────────────────

@info "Plotting TV-metric convergence ..."
set_theme_fonts!(; base_fontsize=16)

fig_tv = Figure(; size = (axis_size()[1], axis_size()[2] * length(HEX_PROBLEMS)))

for (i, prob) in enumerate(HEX_PROBLEMS)
    plot_result_axis!(fig_tv[i, 1], [prob];
        legend = (i == 1),
        metric = :tv,
        plotted_groups = HEX_GROUPS,
        plot_individual_runs = true,
    )
end

save(plot_dir() * "/hex_tv_convergence.png", fig_tv)
save(plot_dir() * "/hex_tv_convergence.pdf", fig_tv)
@info "Saved $(plot_dir())/hex_tv_convergence.png"

# ─── Posterior progression helpers ────────────────────────────────────────────

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

function add_hex_panel!(fig_pos, problem, run_name, run_idx, n_points; grid_size=60)
    ax = Axis(fig_pos;
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )
    prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")
    if !isfile(prob_file)
        text!(ax, 0.5, 0.5; text="no data", align=(:center,:center), space=:relative, fontsize=10, color=:red)
        return ax
    end
    bosip  = load(prob_file)["problem"]
    X_full = bosip.problem.data.X
    Y_full = bosip.problem.data.Y
    N = min(n_points, size(X_full, 2))
    bosip.problem.data = BOSS.ExperimentData(X_full[:, 1:N], Y_full[:, 1:N])
    refit_hex!(bosip)
    log_post_mean = BOSIP.log_posterior_mean(bosip)

    lb, ub = domain(problem).bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect
    XS  = reduce(hcat, [[x1, x2] for x2 in xs2 for x1 in xs1])
    Z   = reshape(exp.(log_post_mean(XS)), grid_size, grid_size)

    heatmap!(ax, xs1, xs2, Z; colormap=:matter)
    scatter!(ax, X_full[1, 1:N], X_full[2, 1:N];
             color=:white, markersize=2, strokewidth=0.3, strokecolor=:black)
    scatter!(ax, [X_full[1, N]], [X_full[2, N]]; color=:red, markersize=4)
    return ax
end

function add_hex_true_panel!(fig_pos, problem; grid_size=60)
    ax = Axis(fig_pos;
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )
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
    heatmap!(ax, xs1, xs2, ys; colormap=:matter)
    x_true = true_params(problem)
    scatter!(ax, [x_true[1]], [x_true[2]]; color=:cyan, marker=:cross, markersize=8, strokewidth=1)
    return ax
end

# ─── Progression grid: rows=problems, cols=groups × snapshots + true ──────────
# Layout: problem label | maxvar@iter5 | maxvar@iter10 | maxvar@iter20 |
#                         eiv@iter5   | eiv@iter10    | eiv@iter20    | true

@info "Plotting data progression ..."

n_points_list = [INIT_DATA_HEX + 5, INIT_DATA_HEX + 10, INIT_DATA_HEX + 20]
snap_labels   = ["iter 5", "iter 10", "iter 20"]
run_idx       = 1  # use the first run for the progression snapshots

ncols = length(HEX_GROUPS) * length(n_points_list) + 1   # +1 for true posterior
nrows = length(HEX_PROBLEMS)
cw, ch = 160, 160

fig_prog = Figure(; size = (cw * ncols + 120, ch * nrows + 40))
colsize!(fig_prog.layout, 1, Fixed(110))

# Column headers (c starts at 2 since col 1 is the label)
for (gi, grp) in enumerate(HEX_GROUPS)
    for (si, lbl) in enumerate(snap_labels)
        c = 1 + (gi - 1) * length(snap_labels) + si
        Label(fig_prog[1, c]; text="$grp\n$lbl", fontsize=11, font=:bold, tellwidth=false)
    end
end
Label(fig_prog[1, ncols + 1]; text="true\nposterior", fontsize=11, font=:bold, tellwidth=false)

# Panels
for (r, prob) in enumerate(HEX_PROBLEMS)
    name = get_name(prob)
    @info "  Row $r: $name ..."
    Label(fig_prog[r+1, 1]; text=name, fontsize=9, tellheight=false, halign=:right)
    for (gi, grp) in enumerate(HEX_GROUPS)
        for (si, np) in enumerate(n_points_list)
            c = 1 + (gi - 1) * length(n_points_list) + si
            add_hex_panel!(fig_prog[r+1, c], prob, grp, run_idx, np)
        end
    end
    add_hex_true_panel!(fig_prog[r+1, ncols + 1], prob)
end

colgap!(fig_prog.layout, 2)
rowgap!(fig_prog.layout, 2)

save(plot_dir() * "/hex_progression.png", fig_prog)
save(plot_dir() * "/hex_progression.pdf", fig_prog)
@info "Done. Saved to $(plot_dir())/hex_progression.png"
