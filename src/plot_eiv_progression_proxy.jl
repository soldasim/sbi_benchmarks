# 24-row × 5-column EIV progression plot — proxy variants substituted.
# BealeProxyProblem and GoldsteinPriceProxyProblem replace the plain versions.
# Proxy problems use _data.jld2 + GP refit (their _problem.jld2 can't be deserialized).

include("main.jl")

using CairoMakie
using JLD2

const INIT_DATA = 3
const FULL_N    = INIT_DATA + 200

all_problems = [
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
    # 2D-only (proxy versions where available)
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
]

@assert length(all_problems) == 24

const PROXY_PROBLEMS = Set(["BealeProxyProblem", "GoldsteinPriceProxyProblem"])

# ─── GP refit helpers ─────────────────────────────────────────────────────────

function make_bosip_from_data(problem, X, Y)
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

function refit!(bosip)
    fitter = OptimizationMAP(;
        algorithm=NEWUOA(), multistart=2, warm_start=true, parallel=false, rhoend=1e-4)
    BOSS.estimate_parameters!(bosip.problem, fitter; options=BossOptions(info=false))
end

# ─── Panels ───────────────────────────────────────────────────────────────────

function add_learned_panel!(fig_pos, problem, run_name, run_idx, n_points; grid_size=60)
    ax = Axis(fig_pos;
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )

    if get_name(problem) in PROXY_PROBLEMS
        fpath = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_data.jld2")
        isfile(fpath) || (text!(ax, 0.5, 0.5; text="no data", align=(:center,:center), space=:relative, fontsize=10, color=:red); return ax)
        X_full, Y_full = load(fpath, "data")
        N = min(n_points, size(X_full, 2))
        bosip = make_bosip_from_data(problem, X_full[:, 1:N], Y_full[:, 1:N])
        refit!(bosip)
        log_post_mean = BOSIP.log_posterior_mean(bosip)
    else
        prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")
        isfile(prob_file) || (text!(ax, 0.5, 0.5; text="no data", align=(:center,:center), space=:relative, fontsize=10, color=:red); return ax)
        bosip  = load(prob_file)["problem"]
        X_full = bosip.problem.data.X
        Y_full = bosip.problem.data.Y
        N = min(n_points, size(X_full, 2))
        bosip.problem.data = BOSS.ExperimentData(X_full[:, 1:N], Y_full[:, 1:N])
        log_post_mean = BOSIP.log_posterior_mean(bosip)
    end

    lb, ub = domain(problem).bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect
    XS  = reduce(hcat, [[x1, x2] for x2 in xs2 for x1 in xs1])
    Z   = reshape(exp.(log_post_mean(XS)), grid_size, grid_size)

    heatmap!(ax, xs1, xs2, Z; colormap=:matter)
    scatter!(ax, X_full[1, 1:N], X_full[2, 1:N];
             color=:white, markersize=2, strokewidth=0.3, strokecolor=:black)
    scatter!(ax, [X_full[1, N]], [X_full[2, N]]; color=:red, markersize=4)

    if N < FULL_N && n_points >= FULL_N
        actual_iters = N - INIT_DATA
        text!(ax, 0.5, 0.03; text="$(actual_iters) iters", align=(:center,:bottom),
              space=:relative, fontsize=9, color=:yellow)
    end
    return ax
end

function add_true_panel!(fig_pos, problem; grid_size=60)
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
    return ax
end

# ─── Build figure ─────────────────────────────────────────────────────────────

mkpath(plot_dir())

col_labels = ["iter 25", "iter 50", "iter 100", "iter 200", "true posterior"]
n_points   = [INIT_DATA+25, INIT_DATA+50, INIT_DATA+100, INIT_DATA+200]
nrows, ncols = length(all_problems), length(col_labels)
cw, ch = 180, 180

@info "Building 24×5 EIV progression plot (proxy substituted) ..."

fig = Figure(; size=(cw*ncols + 120, ch*nrows + 40))
colsize!(fig.layout, 1, Fixed(110))

for (c, lbl) in enumerate(col_labels)
    Label(fig[1, c+1]; text=lbl, fontsize=13, font=:bold, tellwidth=false)
end

for (r, prob) in enumerate(all_problems)
    name = get_name(prob)
    @info "  Row $r: $name ..."
    Label(fig[r+1, 1]; text=name, fontsize=9, tellheight=false, halign=:right)
    for (ci, np) in enumerate(n_points)
        add_learned_panel!(fig[r+1, ci+1], prob, "eiv", 1, np)
    end
    add_true_panel!(fig[r+1, ncols+1], prob)
end

colgap!(fig.layout, 2)
rowgap!(fig.layout, 2)

save(plot_dir() * "/eiv_progression_proxy_grid.png", fig)
save(plot_dir() * "/eiv_progression_proxy_grid.pdf", fig)
@info "Done. Saved to plots/eiv_progression_proxy_grid.png"
