# Plot the true posteriors for the three new optimization-function benchmark problems.
# For d=2: single pairwise marginal heatmap + simulator surface.
# For d>2: full pairwise-marginal matrix (all dim pairs) computed by averaging
#           the true posterior over a Latin Hypercube of the remaining dimensions.

include("main.jl")

using CairoMakie
using Distributions
using LinearAlgebra
using Random

# ---- marginal computation helpers ----

function _lhc_grid(bounds, n)
    lb, ub = bounds
    d = length(lb)
    # Latin hypercube: permute uniform strata per dimension
    X = zeros(d, n)
    for j in 1:d
        perm = randperm(n)
        X[j, :] = lb[j] .+ (ub[j] - lb[j]) .* ((perm .- rand(n)) ./ n)
    end
    return X
end

# Compute normalised pairwise marginal p(x_a, x_b | z) by integrating out
# the remaining dimensions via an LHC average.
function compute_pairwise_marginal(logpost_fn, bounds, dim_a, dim_b;
    grid_size = 60,
    lhc_size  = 300,
)
    lb, ub = bounds
    d = length(lb)
    lhc = _lhc_grid(bounds, lhc_size)   # d × lhc_size

    xs_a = range(lb[dim_a], ub[dim_a]; length=grid_size) |> collect
    xs_b = range(lb[dim_b], ub[dim_b]; length=grid_size) |> collect
    ys   = zeros(grid_size, grid_size)

    # For each (xa, xb) grid point: set those dims in all LHC samples, average exp(logpost)
    col = copy(lhc)
    for (ia, xa) in enumerate(xs_a)
        col[dim_a, :] .= xa
        for (ib, xb) in enumerate(xs_b)
            col[dim_b, :] .= xb
            # compute log-posterior for each LHC column, then average in probability space
            lp = [logpost_fn(col[:, k]) for k in 1:lhc_size]
            lp_max = maximum(lp)
            ys[ia, ib] = mean(exp.(lp .- lp_max)) * exp(lp_max)
        end
    end

    # Normalise to a proper marginal density (trapezoidal rule)
    step_a = (ub[dim_a] - lb[dim_a]) / (grid_size - 1)
    step_b = (ub[dim_b] - lb[dim_b]) / (grid_size - 1)
    total = sum(ys) - 0.5*(sum(ys[1,:]) + sum(ys[end,:]) + sum(ys[:,1]) + sum(ys[:,end]))
    total += 0.25*(ys[1,1] + ys[1,end] + ys[end,1] + ys[end,end])
    (total > 0) && (ys ./= step_a * step_b * total)

    return xs_a, xs_b, ys
end

# Compute normalised 1D marginal for dim_a by integrating out remaining dims.
function compute_diagonal_marginal(logpost_fn, bounds, dim_a;
    grid_size = 200,
    lhc_size  = 300,
)
    lb, ub = bounds
    d = length(lb)
    lhc = _lhc_grid(bounds, lhc_size)

    xs = range(lb[dim_a], ub[dim_a]; length=grid_size) |> collect
    ys = zeros(grid_size)

    col = copy(lhc)
    for (i, xa) in enumerate(xs)
        col[dim_a, :] .= xa
        lp = [logpost_fn(col[:, k]) for k in 1:lhc_size]
        lp_max = maximum(lp)
        ys[i] = mean(exp.(lp .- lp_max)) * exp(lp_max)
    end

    step = (ub[dim_a] - lb[dim_a]) / (grid_size - 1)
    total = sum(ys) - 0.5*(ys[1] + ys[end])
    (total > 0) && (ys ./= step * total)

    return xs, ys
end

# ---- figure builders ----

function plot_2d_posterior(problem::AbstractProblem; grid_size=80, label=nothing)
    @assert x_dim(problem) == 2
    bounds     = domain(problem).bounds
    logpost_fn = true_logpost(problem)
    f          = true_f(problem)
    lb, ub     = bounds

    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect

    # Direct grid evaluation (exact for d=2)
    log_ys = [logpost_fn([x1, x2]) for x1 in xs1, x2 in xs2]
    ys = exp.(log_ys .- maximum(log_ys))
    step1 = (ub[1] - lb[1]) / (grid_size - 1)
    step2 = (ub[2] - lb[2]) / (grid_size - 1)
    total = sum(ys) - 0.5*(sum(ys[1,:]) + sum(ys[end,:]) + sum(ys[:,1]) + sum(ys[:,end]))
    total += 0.25*(ys[1,1] + ys[1,end] + ys[end,1] + ys[end,end])
    (total > 0) && (ys ./= step1 * step2 * total)

    sim_vals = [f([x1, x2])[1] for x1 in xs1, x2 in xs2]

    fig = Figure(; size=(900, 400))
    title_str = isnothing(label) ? get_name(problem) : label

    ax1 = Axis(fig[1,1]; title="Posterior p(x|z)   [$(title_str)]",
        xlabel="x₁", ylabel="x₂")
    hm = heatmap!(ax1, xs1, xs2, ys; colormap=:matter)
    Colorbar(fig[1,2], hm)

    ax2 = Axis(fig[1,3]; title="Simulator f(x)",
        xlabel="x₁", ylabel="x₂")
    hm2 = heatmap!(ax2, xs1, xs2, sim_vals; colormap=:viridis)
    Colorbar(fig[1,4], hm2)

    return fig
end

function plot_hd_posterior(problem::AbstractProblem; grid_size=50, lhc_size=400, label=nothing)
    d      = x_dim(problem)
    @assert d > 2
    bounds     = domain(problem).bounds
    logpost_fn = true_logpost(problem)
    lb, ub     = bounds
    title_str  = isnothing(label) ? get_name(problem) : label

    fig = Figure(; size=(200*d, 200*d))

    # Add title
    fig[0, 1:d] = Label(fig, title_str; font=:bold, fontsize=16)

    param_labels = ["x$i" for i in 1:d]

    # Pre-compute all unique pairs (upper triangle)
    pair_cache = Dict{Tuple{Int,Int}, Tuple}()
    for dim_a in 1:d, dim_b in dim_a+1:d
        @info "  pairwise marginal ($dim_a, $dim_b) ..."
        xs_a, xs_b, ys = compute_pairwise_marginal(logpost_fn, bounds, dim_a, dim_b;
            grid_size, lhc_size)
        pair_cache[(dim_a, dim_b)] = (xs_a, xs_b, ys)
    end

    for dim_a in 1:d, dim_b in 1:d
        if dim_a == dim_b
            @info "  diagonal marginal dim $dim_a ..."
            xs, ys = compute_diagonal_marginal(logpost_fn, bounds, dim_a;
                grid_size=grid_size*2, lhc_size)
            ax = Axis(fig[dim_b, dim_a]; xlabel=param_labels[dim_a])
            lines!(ax, xs, ys)
        else
            lo, hi = min(dim_a, dim_b), max(dim_a, dim_b)
            xs_lo, xs_hi, ys_lh = pair_cache[(lo, hi)]
            if dim_a < dim_b   # lower triangle: xs_a=lo, xs_b=hi
                ax = Axis(fig[dim_b, dim_a]; xlabel=param_labels[dim_a], ylabel=param_labels[dim_b])
                heatmap!(ax, xs_lo, xs_hi, ys_lh; colormap=:matter)
            else               # upper triangle: transposed
                ax = Axis(fig[dim_b, dim_a]; xlabel=param_labels[dim_a], ylabel=param_labels[dim_b])
                heatmap!(ax, xs_hi, xs_lo, ys_lh'; colormap=:matter)
            end
        end
    end

    trim!(fig.layout)
    return fig
end

# ---- main ----

Random.seed!(42)

problems_2d = [
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
]

problems_5d = [
    RosenbrockProblem(; x_dim=5),
    StyblinskiTangProblem(; x_dim=5),
    MichalewiczProblem(; x_dim=5),
]

mkpath(plot_dir())

@info "=== 2D posteriors ==="
for prob in problems_2d
    name = get_name(prob)
    @info "Plotting $name ..."
    fig = plot_2d_posterior(prob; grid_size=100)
    save(plot_dir() * "/opt_posterior_$(name).png", fig)
    save(plot_dir() * "/opt_posterior_$(name).pdf", fig)
    @info "  Saved to $(plot_dir())/opt_posterior_$(name).png"
end

@info "=== 5D posteriors (pairwise marginals) ==="
for prob in problems_5d
    name = get_name(prob)
    @info "Plotting $name ..."
    fig = plot_hd_posterior(prob; grid_size=40, lhc_size=300)
    save(plot_dir() * "/opt_posterior_$(name).png", fig)
    save(plot_dir() * "/opt_posterior_$(name).pdf", fig)
    @info "  Saved to $(plot_dir())/opt_posterior_$(name).png"
end

@info "Done."
