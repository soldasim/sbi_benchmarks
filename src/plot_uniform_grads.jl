# Generate convergence plots for the uniform-grads benchmark.
# Same as plot_uniform.jl but for the gradient-enhanced GP runs.

include("plots.jl")

mkpath(plot_dir())

problems_meangauss = [MeanGauss(; x_dim=d) for d in 1:6]
problems_ab = [MultidimProblem(ABProblem(), d) for d in 1:3]
problems_square = [MultidimProblem(SquareProblem(), d) for d in 1:6]
problems_sine   = [MultidimProblem(SineProblem(),   d) for d in 1:6]
problems_cubic  = [MultidimProblem(CubicProblem(),  d) for d in 1:6]
all_problems = vcat(problems_meangauss, problems_ab, problems_square, problems_sine, problems_cubic)

groups = ["uniform-grads"]

# TV metric (posterior convergence — Thm 5.1/5.2)

# L2 convergence metric (simulator convergence — Thm 3.5)

# Combined grid: all problems, TV metric
@info "Plotting combined TV grid ..."
fig_tv = plot_results(; save_plot=false, metric=:tv, plotted_groups=groups, plot_individual_runs=true)
save(plot_dir() * "/ugrads_tv_all.png", fig_tv)
save(plot_dir() * "/ugrads_tv_all.pdf", fig_tv)

# Combined grid: all problems, convergence metric
@info "Plotting combined convergence grid ..."
fig_conv = plot_results(; save_plot=false, metric=:convergence, plotted_groups=groups, plot_individual_runs=true)
save(plot_dir() * "/ugrads_conv_all.png", fig_conv)
save(plot_dir() * "/ugrads_conv_all.pdf", fig_conv)

@info "All plots saved to $(plot_dir())/"
