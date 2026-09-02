# Plot TV-metric convergence results for the 21 new opt-function benchmark experiments.
# Compares MaxVar vs EIV acquisition at 2D for all new problems.
# Beale and Goldstein-Price use proxy variants (log-transform, CustomLikelihood).

include("plots.jl")

mkpath(plot_dir())

problems_ddim = [
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
]

problems_2d = [
    BealeProxyProblem(),       # proxy variant: log(1+f), same posterior as BealeProblem
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProxyProblem(),  # proxy variant: log(f−2), same posterior as GoldsteinPriceProblem
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
]

all_problems = vcat(problems_ddim, problems_2d)
groups = ["maxvar", "eiv"]

# --- Individual plots for each problem ---
@info "Plotting individual problems ..."
for prob in all_problems
    name = get_name(prob)
    @info "  $name ..."
    fig = plot_results([prob];
        metric = :tv,
        plotted_groups = groups,
        plot_individual_runs = true,
    )
    save(plot_dir() * "/new_opt_results_$(name).png", fig)
    save(plot_dir() * "/new_opt_results_$(name).pdf", fig)
end

# --- Combined grid: d-dimensional problems (3×3) ---
@info "Plotting combined grid (d-dimensional) ..."
set_theme_fonts!(; base_fontsize=18)
nrows, ncols = 3, 3
ax_width, ax_height = axis_size()
fig_ddim = Figure(; size = (ax_width * ncols, ax_height * nrows))
for (i, prob) in enumerate(problems_ddim)
    row = div(i - 1, ncols) + 1
    col = mod(i - 1, ncols) + 1
    plot_result_axis!(fig_ddim[row, col], [prob];
        legend = (i == 1),
        metric = :tv,
        plotted_groups = groups,
        plot_individual_runs = true,
    )
end
save(plot_dir() * "/new_opt_results_ddim_grid.png", fig_ddim)
save(plot_dir() * "/new_opt_results_ddim_grid.pdf", fig_ddim)

# --- Combined grid: 2D-only problems (3×4) ---
@info "Plotting combined grid (2D-only) ..."
nrows2, ncols2 = 3, 4
fig_2d = Figure(; size = (ax_width * ncols2, ax_height * nrows2))
for (i, prob) in enumerate(problems_2d)
    row = div(i - 1, ncols2) + 1
    col = mod(i - 1, ncols2) + 1
    plot_result_axis!(fig_2d[row, col], [prob];
        legend = (i == 1),
        metric = :tv,
        plotted_groups = groups,
        plot_individual_runs = true,
    )
end
save(plot_dir() * "/new_opt_results_2d_grid.png", fig_2d)
save(plot_dir() * "/new_opt_results_2d_grid.pdf", fig_2d)

# --- Single combined grid: all 21 problems (3×7) ---
@info "Plotting single combined grid (all problems) ..."
nrows_all, ncols_all = 3, 7
fig_all = Figure(; size = (ax_width * ncols_all, ax_height * nrows_all))
for (i, prob) in enumerate(all_problems)
    row = div(i - 1, ncols_all) + 1
    col = mod(i - 1, ncols_all) + 1
    plot_result_axis!(fig_all[row, col], [prob];
        legend = (i == 1),
        metric = :tv,
        plotted_groups = groups,
        plot_individual_runs = true,
    )
end
save(plot_dir() * "/new_opt_results_all_grid.png", fig_all)
save(plot_dir() * "/new_opt_results_all_grid.pdf", fig_all)

@info "Done. All plots saved to $(plot_dir())/"
