# TV-metric convergence + pairwise-conditional progression for all 12 cross-polytope 5D problems.
# For each problem the progression shows: TV metric | best MaxVar last-iter | best EIV last-iter | true posterior.
# Pairwise conditionals: lower-triangle 5×5 grid (all dims fixed at true_params except the varying pair).

include("plots.jl")

using CairoMakie
using JLD2

mkpath(plot_dir())

const ALL_CROSS5D_PROBLEMS = CrossPolytopeObsProblem.([
    RosenbrockProblem(; x_dim=5),
    StyblinskiTangProblem(; x_dim=5),
    MichalewiczProblem(; x_dim=5),
    AckleyProblem(; x_dim=5),
    AlpineProblem(; x_dim=5),
    ExpandedSchafferF6Problem(; x_dim=5),
    ExpandedZakharovProblem(; x_dim=5),
    GriewankProblem(; x_dim=5),
    RastriginProblem(; x_dim=5),
    SalomonProblem(; x_dim=5),
    SchwefelProblem(; x_dim=5),
    SphereProblem(; x_dim=5),
])
const CROSS5D_GROUPS = ["maxvar", "eiv"]

@assert length(ALL_CROSS5D_PROBLEMS) == 12

# ─── TV-metric convergence (2×6 grid) ─────────────────────────────────────────

@info "Plotting TV-metric convergence ..."
set_theme_fonts!(; base_fontsize=14)

nrows_tv, ncols_tv = 2, 6
aw, ah = axis_size()
fig_tv = Figure(; size = (aw * ncols_tv, ah * nrows_tv))

for (i, prob) in enumerate(ALL_CROSS5D_PROBLEMS)
    r = div(i - 1, ncols_tv) + 1
    c = mod(i - 1, ncols_tv) + 1
    plot_result_axis!(fig_tv[r, c], [prob];
        legend = (i == 1),
        metric = :tv,
        plotted_groups = CROSS5D_GROUPS,
        plot_individual_runs = true,
        compute_slope = false,
    )
end

save(plot_dir() * "/cross5d_all_tv_convergence.png", fig_tv)
save(plot_dir() * "/cross5d_all_tv_convergence.pdf", fig_tv)
@info "Saved $(plot_dir())/cross5d_all_tv_convergence.png"

# ─── Helpers ──────────────────────────────────────────────────────────────────

function best_run_idx_5d(problem, run_name; n_runs=5)
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

function fill_pairwise_conditionals!(gl, logpost_fn, lb, ub, x_true; grid_size=40, scatter_data=nothing)
    ndim = length(x_true)
    xs = [range(lb[i], ub[i]; length=grid_size) |> collect for i in 1:ndim]

    for row_inner in 1:ndim, col_inner in 1:ndim
        col_inner > row_inner && continue   # upper triangle — leave empty

        ax = Axis(gl[row_inner, col_inner];
            xticklabelsvisible = false,
            yticklabelsvisible = false,
            xticksvisible      = false,
            yticksvisible      = false,
            leftspinevisible   = false,
            rightspinevisible  = false,
            topspinevisible    = false,
            bottomspinevisible = false,
        )

        if row_inner == col_inner
            dim = row_inner
            log_ys = map(xs[dim]) do xi
                x_v = copy(x_true); x_v[dim] = xi
                logpost_fn(x_v)
            end
            ys = exp.(log_ys .- maximum(log_ys))
            lines!(ax, xs[dim], ys; color=:black, linewidth=0.8)
            vlines!(ax, [x_true[dim]]; color=:cyan, linewidth=1.0)
        else
            dim_x = col_inner; dim_y = row_inner
            log_zs = [begin
                x_v = copy(x_true); x_v[dim_x] = xi; x_v[dim_y] = yi
                logpost_fn(x_v)
            end for xi in xs[dim_x], yi in xs[dim_y]]
            zs = exp.(log_zs .- maximum(log_zs))
            heatmap!(ax, xs[dim_x], xs[dim_y], zs; colormap=:matter)
            if !isnothing(scatter_data)
                scatter!(ax, scatter_data[dim_x, :], scatter_data[dim_y, :];
                         color=:white, markersize=2, strokewidth=0.3, strokecolor=:black)
            end
            scatter!(ax, [x_true[dim_x]], [x_true[dim_y]];
                     color=:cyan, marker=:cross, markersize=5, strokewidth=1.0)
        end
    end

    colgap!(gl, 2)
    rowgap!(gl, 2)
end

function add_estimated_pairwise!(parent_layout, row, col, problem, run_name, run_idx; grid_size=40)
    gl = parent_layout[row, col] = GridLayout()

    prob_file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")
    if !isfile(prob_file)
        Label(gl[1, 1]; text="no data", fontsize=8, color=:red, tellwidth=false, tellheight=false)
        return
    end

    try
        bosip      = load(prob_file)["problem"]
        log_post   = BOSIP.log_posterior_mean(bosip)
        X          = bosip.problem.data.X
        lb, ub     = domain(problem).bounds
        x_true     = true_params(problem)
        fill_pairwise_conditionals!(gl, log_post, lb, ub, x_true; grid_size, scatter_data=X)
    catch e
        @warn "Error for $(get_name(problem)) $run_name run $run_idx: $e"
        Label(gl[1, 1]; text="err", fontsize=8, color=:orange, tellwidth=false, tellheight=false)
    end
end

function add_true_pairwise!(parent_layout, row, col, problem; grid_size=40)
    gl = parent_layout[row, col] = GridLayout()
    logpost_fn = true_logpost(problem)
    lb, ub     = domain(problem).bounds
    x_true     = true_params(problem)
    fill_pairwise_conditionals!(gl, logpost_fn, lb, ub, x_true; grid_size)
end

# ─── Progression grid ─────────────────────────────────────────────────────────
# Layout: TV metric | MaxVar best run last iter | EIV best run last iter | true posterior

@info "Plotting progression ..."

ndim       = 5
panel_px   = 80        # px per pairwise panel
tv_cw      = 180       # TV axis width
header_h   = 45
block_size = panel_px * ndim   # 400

fig_prog = Figure()

Label(fig_prog[1, 1]; text="TV metric",                           fontsize=9, font=:bold, tellwidth=false)
Label(fig_prog[1, 2]; text="MaxVar — best run, last iter",        fontsize=9, font=:bold, tellwidth=false)
Label(fig_prog[1, 3]; text="EIV — best run, last iter",           fontsize=9, font=:bold, tellwidth=false)
Label(fig_prog[1, 4]; text="True posterior",                      fontsize=9, font=:bold, tellwidth=false)

for (i, prob) in enumerate(ALL_CROSS5D_PROBLEMS)
    r = i + 1   # +1 for header row
    @info "  Row $i: $(get_name(prob)) ..."

    plot_result_axis!(fig_prog[r, 1], [prob];
        legend = false, metric = :tv,
        plotted_groups = CROSS5D_GROUPS,
        plot_individual_runs = true,
        compute_slope = false,
    )

    maxvar_idx = best_run_idx_5d(prob, "maxvar")
    eiv_idx    = best_run_idx_5d(prob, "eiv")

    add_estimated_pairwise!(fig_prog.layout, r, 2, prob, "maxvar", maxvar_idx; grid_size=40)
    add_estimated_pairwise!(fig_prog.layout, r, 3, prob, "eiv",    eiv_idx;    grid_size=40)
    add_true_pairwise!(     fig_prog.layout, r, 4, prob;                       grid_size=40)
end

colsize!(fig_prog.layout, 1, Fixed(tv_cw))
for c in 2:4
    colsize!(fig_prog.layout, c, Fixed(block_size))
end
rowsize!(fig_prog.layout, 1, Fixed(header_h))
for r in 1:length(ALL_CROSS5D_PROBLEMS)
    rowsize!(fig_prog.layout, r + 1, Fixed(block_size))
end
colgap!(fig_prog.layout, 6)
rowgap!(fig_prog.layout, 4)
resize_to_layout!(fig_prog)
w, h = size(fig_prog.scene)
@info "Figure size: $(w) × $(h) pts"

save(plot_dir() * "/cross5d_all_progression_best.png", fig_prog)
save(plot_dir() * "/cross5d_all_progression_best.pdf", fig_prog)
@info "Done. Saved to $(plot_dir())/cross5d_all_progression_best.png"
