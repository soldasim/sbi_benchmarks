# Plot empirical convergence slopes vs dimension for the uniform benchmark,
# with theoretical prediction from Teckentrup Thm 3.5.
# Theory: Matérn-5/2 (ν=2.5), infinitely smooth functions → γ_0 = ν/(2ν+d)

include("plots.jl")

function compute_slope_for_problem(problem::AbstractProblem, group::String; metric::Symbol=:tv)
    scores_by_group = if metric == :convergence
        load_stored_convergence_scores([problem])
    else
        load_stored_scores([problem], TVMetric)
    end

    key = get_name(problem) * "_" * group
    haskey(scores_by_group, key) || return nothing

    scores = scores_by_group[key]
    isempty(scores) && return nothing

    if !allequal(length.(scores))
        scores = align_scores(scores)
    end
    arr = reduce(hcat, scores)
    maxlen = size(arr, 1)
    xs = 3:3+maxlen-1

    median_scores = mapslices(median∘skipmissing, arr; dims=2)[:]
    any(isnan, median_scores) && return nothing

    slope, _, _, _, _, _ = compute_convergence_slope(collect(xs), median_scores)
    return slope
end

function plot_slopes_vs_dimension(; metric::Symbol=:tv, save_plot=false, base_fontsize=20, ν=2.5)
    set_theme_fonts!(base_fontsize=base_fontsize)

    # Problem families: (label, color, marker, problems, dims)
    families = [
        ("MeanGauss",   :blue,   :circle, [MeanGauss(; x_dim=d) for d in 1:6],              collect(1:6)),
        ("MultidimAB",  :orange, :rect,   [MultidimProblem(ABProblem(), s) for s in 1:3],    [2,4,6]),
        ("Square",      :green,  :diamond,[MultidimProblem(SquareProblem(), d) for d in 1:6],collect(1:6)),
        ("Sine",        :red,    :utriangle,[MultidimProblem(SineProblem(), d) for d in 1:6],collect(1:6)),
        ("Cubic",       :purple, :dtriangle,[MultidimProblem(CubicProblem(), d) for d in 1:6],collect(1:6)),
    ]

    metric_label = metric == :convergence ? "L2 risk (simulator)" : "TV metric (posterior)"

    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel = "Dimension d",
        ylabel = "|slope| in log-log space",
        title  = "Convergence slope vs dimension\n($metric_label, uniform acquisition)",
    )

    all_dims = Int[]
    for (label, color, marker, problems, dims) in families
        slopes = [compute_slope_for_problem(p, "uniform"; metric) for p in problems]
        valid = [(d, s) for (d, s) in zip(dims, slopes) if !isnothing(s)]
        isempty(valid) && continue
        append!(all_dims, first.(valid))
        scatter!(ax, first.(valid), abs.(last.(valid));
            label=label, markersize=12, color=color, marker=marker)
    end

    if !isempty(all_dims)
        d_theory = range(minimum(all_dims) - 0.3, maximum(all_dims) + 0.3; length=200)
        lines!(ax, d_theory, ν ./ (2ν .+ d_theory);
            label="Theory: ν/(2ν+d), ν=$(ν)", linestyle=:dash, linewidth=2, color=:black)
    end

    axislegend(ax; position=:rt)

    if save_plot
        mkpath(plot_dir())
        save(plot_dir() * "/slopes_vs_dimension_$(metric).png", fig)
        save(plot_dir() * "/slopes_vs_dimension_$(metric).pdf", fig)
        @info "Saved to $(plot_dir())/slopes_vs_dimension_$(metric).png"
    end
    return fig
end

plot_slopes_vs_dimension(; metric=:tv,          save_plot=true)
plot_slopes_vs_dimension(; metric=:convergence,  save_plot=true)
@info "Done."
