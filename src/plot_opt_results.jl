# Plot TV-metric convergence results for the opt-function benchmark experiments.
# Compares MaxVar vs EIV acquisition at d=2 and d=5 for Rosenbrock, StyblinskiTang, Michalewicz.

include("plots.jl")

mkpath(plot_dir())

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
all_problems = vcat(problems_2d, problems_5d)

groups = ["maxvar", "eiv"]

# --- Combined grid: 2 rows × 3 cols (2D top, 5D bottom) ---
@info "Plotting combined grid ..."
set_theme_fonts!(; base_fontsize=18)

problems_grid = reshape(all_problems, 2, 3)  # [d=2; d=5] × [Rosen, Styb, Mich]
nrows, ncols = size(problems_grid)
ax_width, ax_height = axis_size()
fig = Figure(; size = (ax_width * ncols, ax_height * nrows))

for idx in CartesianIndices(problems_grid)
    plot_result_axis!(fig[idx.I...], [problems_grid[idx]];
        legend = (idx == CartesianIndex(1, 1)),
        metric = :tv,
        plotted_groups = groups,
        plot_individual_runs = true,
    )
end

save(plot_dir() * "/opt_results_tv_grid.png", fig)
save(plot_dir() * "/opt_results_tv_grid.pdf", fig)
@info "Saved to $(plot_dir())/opt_results_tv_grid.png"

# --- Individual problem plots ---
@info "Plotting individual problems ..."
for prob in all_problems
    name = get_name(prob)
    @info "  $name ..."
    fig_ind = plot_results([prob];
        metric = :tv,
        plotted_groups = groups,
        plot_individual_runs = true,
    )
    save(plot_dir() * "/opt_results_$(name).png", fig_ind)
    save(plot_dir() * "/opt_results_$(name).pdf", fig_ind)
end

@info "Done."
