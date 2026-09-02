# TV-metric convergence + learned-posterior progression for all 24 hex opt-function problems.
# Run from the repo root after experiments have finished.

include("plots.jl")

using CairoMakie
using JLD2

mkpath(plot_dir())

const INIT_DATA_HEX = 3
const ALL_HEX_PROBLEMS = HexObsProblem.([
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
const HEX_GROUPS = ["maxvar", "eiv"]

@assert length(ALL_HEX_PROBLEMS) == 24

# ─── TV-metric convergence (4×6 grid) ─────────────────────────────────────────

@info "Plotting TV-metric convergence ..."
set_theme_fonts!(; base_fontsize=14)

nrows_tv, ncols_tv = 4, 6
aw, ah = axis_size()
fig_tv = Figure(; size = (aw * ncols_tv, ah * nrows_tv))

for (i, prob) in enumerate(ALL_HEX_PROBLEMS)
    r = div(i - 1, ncols_tv) + 1
    c = mod(i - 1, ncols_tv) + 1
    plot_result_axis!(fig_tv[r, c], [prob];
        legend = (i == 1),
        metric = :tv,
        plotted_groups = HEX_GROUPS,
        plot_individual_runs = true,
        compute_slope = false,
    )
end

save(plot_dir() * "/hex_all_tv_convergence.png", fig_tv)
save(plot_dir() * "/hex_all_tv_convergence.pdf", fig_tv)
@info "Saved $(plot_dir())/hex_all_tv_convergence.png"

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

function add_hex_panel!(fig_pos, problem, run_name, run_idx, n_points; grid_size=50)
    ax = Axis(fig_pos;
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )
    prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")
    if !isfile(prob_file)
        text!(ax, 0.5, 0.5; text="no data", align=(:center,:center), space=:relative, fontsize=8, color=:red)
        return ax
    end
    bosip  = load(prob_file)["problem"]
    X_full = bosip.problem.data.X
    Y_full = bosip.problem.data.Y
    N = min(n_points, size(X_full, 2))
    bosip.problem.data = BOSS.ExperimentData(X_full[:, 1:N], Y_full[:, 1:N])
    try
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
        scatter!(ax, [X_full[1, N]], [X_full[2, N]]; color=:red, markersize=3)
        if size(X_full, 2) < n_points
            actual_iters = size(X_full, 2) - INIT_DATA_HEX
            text!(ax, 0.5, 0.02; text="$actual_iters iters", align=(:center,:bottom),
                  space=:relative, fontsize=10, color=:black)
        end
    catch e
        @warn "Panel error for $(get_name(problem)) $run_name@$n_points: $e"
        text!(ax, 0.5, 0.5; text="err", align=(:center,:center), space=:relative, fontsize=8, color=:orange)
    end
    return ax
end

function add_hex_true_panel!(fig_pos, problem; grid_size=50)
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
    scatter!(ax, [x_true[1]], [x_true[2]]; color=:cyan, marker=:cross, markersize=6, strokewidth=1)
    return ax
end

# ─── Progression grid: rows=problems, cols=groups × snapshots + true ──────────
# Layout: problem label | maxvar@iter50 | maxvar@iter100 | maxvar@iter200 |
#                         eiv@iter50   | eiv@iter100    | eiv@iter200    | true

@info "Plotting data progression ..."

n_points_list = [INIT_DATA_HEX + 50, INIT_DATA_HEX + 100, INIT_DATA_HEX + 200]
snap_labels   = ["iter 50", "iter 100", "iter 200"]

function best_run_idx(problem, run_name; n_runs=5)
    dir = data_dir(problem)
    best_idx = 1
    best_score = Inf
    for idx in 1:n_runs
        file = joinpath(dir, "$(run_name)_$(idx)_TVmetric.jld2")
        isfile(file) || continue
        scores = load(file, "score")
        isempty(scores) && continue
        final = scores[end]
        if final < best_score
            best_score = final
            best_idx = idx
        end
    end
    return best_idx
end

ncols = length(HEX_GROUPS) * length(n_points_list) + 1   # +1 for true posterior
ch = 180   # 1.5x the original 120

tv_cw = 150  # narrower TV metric column
header_h = 50
col_gap = 2
row_gap = 2

function build_progression_fig(problems)
    nrows = length(problems)
    fig = Figure()

    Label(fig[1, 1]; text="TV metric", fontsize=9, font=:bold, tellwidth=false)
    for (gi, grp) in enumerate(HEX_GROUPS)
        for (si, lbl) in enumerate(snap_labels)
            c = 1 + (gi - 1) * length(snap_labels) + si
            Label(fig[1, c]; text="$grp\n$lbl", fontsize=9, font=:bold, tellwidth=false)
        end
    end
    Label(fig[1, ncols + 1]; text="true\nposterior", fontsize=9, font=:bold, tellwidth=false)

    for (r, prob) in enumerate(problems)
        name = get_name(prob)
        @info "  Row $r: $name ..."
        plot_result_axis!(fig[r+1, 1], [prob];
            legend = false, metric = :tv,
            plotted_groups = HEX_GROUPS,
            plot_individual_runs = true,
            compute_slope = false,
        )
        for (gi, grp) in enumerate(HEX_GROUPS)
            ridx = best_run_idx(prob, grp)
            for (si, np) in enumerate(n_points_list)
                c = 1 + (gi - 1) * length(n_points_list) + si
                add_hex_panel!(fig[r+1, c], prob, grp, ridx, np)
            end
        end
        add_hex_true_panel!(fig[r+1, ncols + 1], prob)
    end

    colsize!(fig.layout, 1, Fixed(tv_cw))
    for c in 2:(ncols + 1)
        colsize!(fig.layout, c, Fixed(ch))
    end
    rowsize!(fig.layout, 1, Fixed(header_h))
    for r in 1:nrows
        rowsize!(fig.layout, r + 1, Fixed(ch))
    end
    colgap!(fig.layout, col_gap)
    rowgap!(fig.layout, row_gap)
    # Let Makie compute the true figure size including axis protrusions and padding
    resize_to_layout!(fig)
    w, h = size(fig.scene)
    @info "Figure size after resize_to_layout!: $(w) × $(h) pts"
    return fig
end

@info "Building full 24-row progression plot ..."
fig = build_progression_fig(ALL_HEX_PROBLEMS)
save(plot_dir() * "/hex_all_progression_best.png", fig)
save(plot_dir() * "/hex_all_progression_best.pdf", fig)
@info "Done. Saved to $(plot_dir())/hex_all_progression_best.png"
