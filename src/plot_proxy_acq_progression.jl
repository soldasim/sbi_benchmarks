# 4-row × 5-column progression plot comparing MaxVar and EIV for proxy problems.
# Rows: BealeProxy (MaxVar), BealeProxy (EIV), GoldsteinPriceProxy (MaxVar), GoldsteinPriceProxy (EIV)
# Cols: learned posterior at iter 25 | 50 | 100 | 200 | true posterior
#
# Uses _data.jld2 + GP refit (proxy _problem.jld2 can't be deserialized).
#
# Run from: ~/repos/bosip_benchmarks/
#   julia --project=src src/plot_proxy_acq_progression.jl

include("main.jl")

using CairoMakie
using JLD2

const INIT_DATA = 3
const FULL_N    = INIT_DATA + 200

# ─── GP refit helpers (proxy problems have anonymous fns, can't load _problem.jld2) ──

function make_bosip(problem::AbstractProblem, X::AbstractMatrix, Y::AbstractMatrix)
    data  = BOSS.ExperimentData(X, Y)
    model = GaussianProcess(;
        mean               = prior_mean(problem),
        kernel             = BOSS.Matern52Kernel(),
        lengthscale_priors = get_lengthscale_priors(problem),
        amplitude_priors   = get_amplitude_priors(problem),
        noise_std_priors   = get_noise_std_priors(problem),
    )
    return construct_bosip_problem(;
        problem,
        data,
        acquisition = LogMaxVar(),
        model,
    )
end

function fit_bosip!(bosip::BosipProblem)
    fitter = OptimizationMAP(;
        algorithm  = NEWUOA(),
        multistart = 2,
        warm_start = true,
        parallel   = false,
        rhoend     = 1e-4,
    )
    BOSS.estimate_parameters!(bosip.problem, fitter; options=BossOptions(info=false))
end

# ─── Learned posterior panel ───────────────────────────────────────────────────

function add_learned_panel!(fig_pos, problem::AbstractProblem,
                             run_name::String, run_idx::Int, n_points::Int;
                             grid_size=60)
    ax = Axis(fig_pos;
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )

    fpath = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_data.jld2")
    if !isfile(fpath)
        text!(ax, 0.5, 0.5; text="no data", align=(:center, :center),
              space=:relative, fontsize=10, color=:red)
        return ax
    end

    X_full, Y_full = load(fpath, "data")
    N_avail = size(X_full, 2)
    N       = min(n_points, N_avail)

    bosip = make_bosip(problem, X_full[:, 1:N], Y_full[:, 1:N])
    fit_bosip!(bosip)
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

    if N_avail < FULL_N && n_points >= FULL_N
        actual_iters = N_avail - INIT_DATA
        text!(ax, 0.5, 0.03; text="$(actual_iters) iters", align=(:center, :bottom),
              space=:relative, fontsize=9, color=:yellow)
    end

    return ax
end

# ─── True posterior panel ──────────────────────────────────────────────────────

function add_true_panel!(fig_pos, problem::AbstractProblem; grid_size=60)
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

# ─── Row definitions ──────────────────────────────────────────────────────────
# (problem, run_name, run_idx, row_label)

rows = [
    (BealeProxyProblem(),          "maxvar", 1, "BealeProxy\nMaxVar"),
    (BealeProxyProblem(),          "eiv",    1, "BealeProxy\nEIV"),
    (GoldsteinPriceProxyProblem(), "maxvar", 1, "GoldsteinPriceProxy\nMaxVar"),
    (GoldsteinPriceProxyProblem(), "eiv",    1, "GoldsteinPriceProxy\nEIV"),
]

col_labels = ["iter 25", "iter 50", "iter 100", "iter 200", "true posterior"]
n_points   = [INIT_DATA+25, INIT_DATA+50, INIT_DATA+100, INIT_DATA+200]

# ─── Build figure ─────────────────────────────────────────────────────────────

mkpath(plot_dir())

nrows       = length(rows)
ncols       = length(col_labels)
cw, ch      = 200, 200
label_col_w = 130

@info "Building $(nrows)×$(ncols) proxy acquisition progression plot ..."

fig = Figure(; size = (cw*ncols + label_col_w + 10, ch*nrows + 40))
colsize!(fig.layout, 1, Fixed(label_col_w))

for (c, lbl) in enumerate(col_labels)
    Label(fig[1, c+1]; text=lbl, fontsize=13, font=:bold, tellwidth=false)
end

for (r, (prob, run_name, run_idx, row_label)) in enumerate(rows)
    @info "  Row $r: $(get_name(prob)) / $run_name ..."

    Label(fig[r+1, 1]; text=row_label, fontsize=10, halign=:right,
          tellheight=false, justification=:right)

    for (ci, np) in enumerate(n_points)
        @info "    col $ci ..."
        add_learned_panel!(fig[r+1, ci+1], prob, run_name, run_idx, np)
    end

    add_true_panel!(fig[r+1, ncols+1], prob)
end

colgap!(fig.layout, 4)
rowgap!(fig.layout, 4)

fname_base = plot_dir() * "/proxy_acq_progression"
save(fname_base * ".png", fig)
save(fname_base * ".pdf", fig)
@info "Done. Saved to $(fname_base).png"
