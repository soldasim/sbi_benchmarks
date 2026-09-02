# 6-row × 5-column progression plot: without/with proxy for SIR, Beale, Goldstein-Price.
# Rows (alternating without/with proxy):
#   1. SIR (no proxy)           — data-bosip/SIRProblem,               method: standard
#   2. SIR (proxy)              — data-bosip/ProxySIRProblem,          method: standard
#   3. Beale (no proxy)         — data-opt-functions/BealeProblem,      method: maxvar
#   4. Beale (proxy)            — data-opt-functions/BealeProxyProblem, method: maxvar
#   5. Goldstein-Price (no proxy) — data-opt-functions/GoldsteinPriceProblem
#   6. Goldstein-Price (proxy)    — data-opt-functions/GoldsteinPriceProxyProblem
#
# Columns 1-4: GP posterior mean heatmap (re-fitted from saved X,Y) + data scatter
#              at iter 25 / 50 / 100 / 200.
# Column 5:   true posterior heatmap.
# If a run has fewer than 200 BO iters, the iter-200 column uses the maximum available
# and annotates accordingly.
#
# Run from: ~/repos/bosip_benchmarks/
#   julia --project=src src/plot_proxy_progression.jl

include("main.jl")

using CairoMakie
using JLD2

const PLOT_DIR = "plots"
const INIT_DATA = 3

# ─────────────────────────────────────────────
# Build & fit a BosipProblem from raw X, Y
# ─────────────────────────────────────────────

function make_bosip(problem::AbstractProblem, X::AbstractMatrix, Y::AbstractMatrix)
    data = BOSS.ExperimentData(X, Y)
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

# ─────────────────────────────────────────────
# Learned-posterior panel
# ─────────────────────────────────────────────

function add_learned_panel!(fig_pos, problem::AbstractProblem, data_root::String,
                             run_name::String, run_idx::Int,
                             n_points::Int, max_avail::Int,
                             col_label::String; grid_size=60)
    actual_n  = min(n_points, max_avail)
    truncated = (actual_n < n_points)
    title_str = truncated ? "iter $(actual_n - INIT_DATA) (max)" : col_label

    ax = Axis(fig_pos;
        title     = title_str,
        titlesize = 11,
        xticklabelsvisible = false, yticklabelsvisible = false,
        xticksvisible      = false, yticksvisible      = false,
        leftspinevisible   = false, rightspinevisible   = false,
        topspinevisible    = false, bottomspinevisible  = false,
    )

    fpath = joinpath(data_root, get_name(problem), "$(run_name)_$(run_idx)_data.jld2")
    if !isfile(fpath)
        text!(ax, 0.5, 0.5; text="no data", align=(:center, :center),
              space=:relative, fontsize=10, color=:red)
        return ax
    end

    X_full, Y_full = load(fpath, "data")
    N = actual_n

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
    return ax
end

# ─────────────────────────────────────────────
# True-posterior panel
# ─────────────────────────────────────────────

function add_true_panel!(fig_pos, problem::AbstractProblem; grid_size=80)
    ax = Axis(fig_pos;
        title     = "true posterior",
        titlesize = 11,
        xticklabelsvisible = false, yticklabelsvisible = false,
        xticksvisible      = false, yticksvisible      = false,
        leftspinevisible   = false, rightspinevisible   = false,
        topspinevisible    = false, bottomspinevisible  = false,
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

# ─────────────────────────────────────────────
# Row definitions
# ─────────────────────────────────────────────

# (problem, data_root, run_name, run_idx, max_avail_evals, row_label)
rows = [
    (SIRProblem(),                 "data-bosip",        "standard", 1, 103, "SIR\n(no proxy)"),
    (ProxySIRProblem(),            "data-bosip",        "standard", 1, 103, "SIR\n(proxy)"),
    (BealeProblem(),               "data-opt-functions", "maxvar",   1, 203, "Beale\n(no proxy)"),
    (BealeProxyProblem(),          "data-opt-functions", "maxvar",   1, 203, "Beale\n(proxy)"),
    (GoldsteinPriceProblem(),      "data-opt-functions", "maxvar",   1, 203, "Goldstein-Price\n(no proxy)"),
    (GoldsteinPriceProxyProblem(), "data-opt-functions", "maxvar",   1, 203, "Goldstein-Price\n(proxy)"),
]

snap_n_points = [INIT_DATA+25, INIT_DATA+50, INIT_DATA+100, INIT_DATA+200]
col_labels    = ["iter 25", "iter 50", "iter 100", "iter 200"]

# ─────────────────────────────────────────────
# Build figure
# ─────────────────────────────────────────────

mkpath(PLOT_DIR)

nrows       = length(rows)
ncols_data  = length(snap_n_points)
ncols_total = ncols_data + 1
cw, ch      = 190, 200
label_col_w = 80

@info "Building 6×5 proxy progression plot ..."

fig = Figure(; size = (cw*ncols_total + label_col_w + 10, ch*nrows + 40))
colsize!(fig.layout, 1, Fixed(label_col_w))

Label(fig[0, ncols_data+2]; text="true posterior", fontsize=12, font=:bold, tellwidth=false)

for (r, (prob, droot, run_name, run_idx, max_avail, row_label)) in enumerate(rows)
    @info "  Row $r: $(get_name(prob)) ..."

    Label(fig[r, 1]; text=row_label, fontsize=10, halign=:right,
          tellheight=false, justification=:right)

    for (ci, np) in enumerate(snap_n_points)
        @info "    col $ci ($(col_labels[ci])) ..."
        add_learned_panel!(fig[r, ci+1], prob, droot, run_name, run_idx,
                           np, max_avail, col_labels[ci])
    end

    add_true_panel!(fig[r, ncols_data+2], prob)
end

colgap!(fig.layout, 4)
rowgap!(fig.layout, 4)

save(joinpath(PLOT_DIR, "proxy_progression.png"), fig)
save(joinpath(PLOT_DIR, "proxy_progression.pdf"), fig)
@info "Done. Saved to $(PLOT_DIR)/proxy_progression.png"
