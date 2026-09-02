# 4×6 grid of simulator output histograms for all 24 opt-function problems.
# Samples from the prior, evaluates f(x), plots histogram with z_obs marked.

include("main.jl")

using CairoMakie
using Distributions
using Random

Random.seed!(42)

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

const N_SAMPLES = 10_000

function add_histogram_axis!(fig_pos, problem::AbstractProblem)
    f      = true_f(problem)
    prior  = x_prior(problem)
    z_obs  = prior_mean(problem)[1]

    xs = rand(prior, N_SAMPLES)
    ys = [f(xs[:, i])[1] for i in 1:N_SAMPLES]

    ax = Axis(fig_pos;
        title      = get_name(problem),
        xlabel     = "f(x)", ylabel = "density",
        titlesize  = 13,
        xlabelsize = 10, ylabelsize = 10,
        xticklabelsize = 8, yticklabelsize = 8,
    )

    hist!(ax, ys; normalization=:pdf, bins=50, color=(:steelblue, 0.7))
    vlines!(ax, [z_obs]; color=:red, linewidth=2, label="z_obs")

    return ax
end

mkpath(plot_dir())

@info "Plotting 4×6 grid of simulator output histograms ..."

nrows, ncols = 4, 6
fig = Figure(; size = (250*ncols, 200*nrows))

for (i, prob) in enumerate(all_problems)
    row = div(i - 1, ncols) + 1
    col = mod(i - 1, ncols) + 1
    @info "  [$row,$col] $(get_name(prob)) ..."
    add_histogram_axis!(fig[row, col], prob)
end

fname = plot_dir() * "/all_opt_simulator_histograms.png"
save(fname, fig)
save(plot_dir() * "/all_opt_simulator_histograms.pdf", fig)
@info "Done. Saved to $fname"
