# 24-row × 4-column progression plot for MaxVar run 1.
# Rows: problems (same order as all other grids)
# Cols: learned posterior at iter 50 | 100 | 200 | true posterior

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

# --- Learned posterior panel (uses truncated data + final hyperparams) ---

function add_learned_panel!(fig_pos, problem::AbstractProblem, run_name::String, run_idx::Int, n_points::Int; grid_size=60)
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
    N      = min(n_points, size(X_full, 2))

    bosip.problem.data = BOSS.ExperimentData(X_full[:, 1:N], Y_full[:, 1:N])
    log_post_mean = BOSIP.log_posterior_mean(bosip)

    lb, ub = domain(problem).bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect
    XS  = reduce(hcat, [x1, x2] for x2 in xs2 for x1 in xs1)
    Z   = reshape(exp.(log_post_mean(XS)), grid_size, grid_size)

    heatmap!(ax, xs1, xs2, Z; colormap=:matter)
    scatter!(ax, X_full[1, 1:N], X_full[2, 1:N];
        color=:white, markersize=2, strokewidth=0.3, strokecolor=:black)
    scatter!(ax, [X_full[1, N]], [X_full[2, N]]; color=:red, markersize=4)
    return ax
end

# --- True posterior panel ---

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
    total = sum(ys) - 0.5*(sum(ys[1,:]) + sum(ys[end,:]) + sum(ys[:,1]) + sum(ys[:,end]))
    total += 0.25*(ys[1,1] + ys[1,end] + ys[end,1] + ys[end,end])
    (total > 0) && (ys ./= step1 * step2 * total)

    heatmap!(ax, xs1, xs2, ys; colormap=:matter)
    return ax
end

# --- Build figure ---

mkpath(plot_dir())

nrows  = length(all_problems)  # 24
ncols  = 5
cw, ch = 180, 180              # cell width / height in px
col_labels = ["iter 25", "iter 50", "iter 100", "iter 200", "true posterior"]
n_points   = [3+25, 3+50, 3+100, 3+200]  # data counts for learned cols

@info "Building 24×5 MaxVar progression plot ..."

fig = Figure(; size = (cw*ncols + 120, ch*nrows + 40))
colsize!(fig.layout, 1, Fixed(110))  # narrow label column

# Column headers
for (c, lbl) in enumerate(col_labels)
    Label(fig[1, c+1]; text=lbl, fontsize=13, font=:bold, tellwidth=false)
end

# Row label + panels
for (r, prob) in enumerate(all_problems)
    name = get_name(prob)
    @info "  Row $r: $name ..."

    # Row label in column 1
    Label(fig[r+1, 1]; text=name, fontsize=9, rotation=0,
          tellheight=false, halign=:right)

    # Learned posteriors (cols 2 to 2+length(n_points)-1)
    for (ci, np) in enumerate(n_points)
        add_learned_panel!(fig[r+1, ci+1], prob, "maxvar", 1, np)
    end

    # True posterior (one column after the last learned panel)
    add_true_panel!(fig[r+1, length(n_points)+2], prob)
end

colgap!(fig.layout, 2)
rowgap!(fig.layout, 2)

fname = plot_dir() * "/maxvar_progression_grid.png"
save(fname, fig)
save(plot_dir() * "/maxvar_progression_grid.pdf", fig)
@info "Done. Saved to $fname"
