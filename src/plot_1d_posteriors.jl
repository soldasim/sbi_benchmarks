# Plot the true 1D posteriors for the three new analytical problems.

include("main.jl")

using CairoMakie
using Distributions

function plot_1d_posterior(problem::AbstractProblem; n_points=1000)
    lb = domain(problem).bounds[1][1]
    ub = domain(problem).bounds[2][1]
    xs = range(lb, ub; length=n_points)

    logpost_fn = true_logpost(problem)
    logpost_vals = [logpost_fn([x]) for x in xs]

    # Normalise via trapezoidal integration
    post_vals = exp.(logpost_vals .- maximum(logpost_vals))
    Z = sum((post_vals[1:end-1] .+ post_vals[2:end]) ./ 2 .* step(xs))
    post_vals ./= Z

    # Simulator curve
    f = true_f(problem)
    sim_vals = [f([x])[1] for x in xs]

    return collect(xs), post_vals, sim_vals
end

problems = [SquareProblem(), SineProblem(), CubicProblem()]
titles   = ["SquareProblem  (y = x²,  z = 1.0, σ = 0.2)",
            "SineProblem    (y = sin(x),  z = 0.7, σ = 0.1)",
            "CubicProblem   (y = x³−x,  z = 0.0, σ = 0.1)"]

fig = Figure(; size=(1100, 900))

for (i, (prob, title)) in enumerate(zip(problems, titles))
    xs, post, sim = plot_1d_posterior(prob)

    # Posterior panel
    ax_post = Axis(fig[i, 1];
        title  = title,
        xlabel = "x",
        ylabel = "posterior p(x|z)",
    )
    lines!(ax_post, xs, post; color=:blue, linewidth=2, label="posterior")
    axislegend(ax_post; position=:rt)

    # Simulator panel
    lb = domain(prob).bounds[1][1]
    ub = domain(prob).bounds[2][1]
    z_obs = likelihood(prob).z_obs[1]
    ax_sim = Axis(fig[i, 2];
        title  = "Simulator f(x)",
        xlabel = "x",
        ylabel = "y = f(x)",
    )
    lines!(ax_sim, xs, sim; color=:green, linewidth=2, label="f(x)")
    hlines!(ax_sim, [z_obs]; color=:red, linestyle=:dash, linewidth=1.5, label="z_obs")
    axislegend(ax_sim; position=:rt)
end

mkpath(plot_dir())
save(plot_dir() * "/analytical1d_posteriors.png", fig)
save(plot_dir() * "/analytical1d_posteriors.pdf", fig)
@info "Saved to $(plot_dir())/analytical1d_posteriors.png"
