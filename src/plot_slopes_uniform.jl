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

    fig = Figure(; size=(1000, 500))
    ax = Axis(fig[1, 1];
        xlabel = "Dimension d",
        ylabel = "|slope| in log-log space",
        title  = "Convergence slope vs dimension\n($metric_label, uniform acquisition)",
    )

    all_dims = Int[]
    for (label, color, marker, problems, dims) in families
        for (group, linestyle) in [("uniform", :solid), ("uniform-grads", :dash)]
            slopes = [compute_slope_for_problem(p, group; metric) for p in problems]
            valid = [(d, s) for (d, s) in zip(dims, slopes) if !isnothing(s)]
            isempty(valid) && continue
            append!(all_dims, first.(valid))
            group_label = group == "uniform" ? label : label * " (grads)"
            scatterlines!(ax, first.(valid), abs.(last.(valid));
                label=group_label, markersize=12, color=color, marker=marker,
                linewidth=2, linestyle=linestyle)
        end
    end

    if !isempty(all_dims)
        d_theory = range(minimum(all_dims) - 0.3, maximum(all_dims) + 0.3; length=200)
        lines!(ax, d_theory, ν ./ (2ν .+ d_theory);
            label="Theory: ν/(2ν+d), ν=$(ν)", linestyle=:dash, linewidth=2, color=:black)
    end

    Legend(fig[1, 2], ax)

    if save_plot
        mkpath(plot_dir())
        save(plot_dir() * "/slopes_vs_dimension_$(metric).png", fig)
        save(plot_dir() * "/slopes_vs_dimension_$(metric).pdf", fig)
        @info "Saved to $(plot_dir())/slopes_vs_dimension_$(metric).png"
    end
    return fig
end

function median_scores_for_group(problem::AbstractProblem, group::String; metric::Symbol=:tv)
    scores_by_group = if metric == :convergence
        load_stored_convergence_scores([problem])
    else
        load_stored_scores([problem], TVMetric)
    end
    key = get_name(problem) * "_" * group
    haskey(scores_by_group, key) || return nothing, nothing
    scores = scores_by_group[key]
    isempty(scores) && return nothing, nothing
    if !allequal(length.(scores))
        scores = align_scores(scores)
    end
    arr = reduce(hcat, scores)
    maxlen = size(arr, 1)
    xs = collect(3:3+maxlen-1)
    ys = mapslices(median∘skipmissing, arr; dims=2)[:]
    any(isnan, ys) && return nothing, nothing
    return xs, ys
end

function fit_loglog_line(xs::AbstractVector, ys::AbstractVector, x_start, x_end)
    idxs = findall(x -> x_start <= x <= x_end, xs)
    isempty(idxs) && return nothing, nothing
    log_xs = log10.(xs[idxs])
    log_ys = log10.(ys[idxs])
    mean_x = mean(log_xs)
    mean_y = mean(log_ys)
    slope = sum((log_xs .- mean_x) .* (log_ys .- mean_y)) / sum((log_xs .- mean_x) .^ 2)
    intercept = mean_y - slope * mean_x
    return slope, intercept  # log10(y) = slope * log10(x) + intercept
end

function compute_logratio_gap_for_problem(problem::AbstractProblem; metric::Symbol=:tv)
    xs_std,   ys_std   = median_scores_for_group(problem, "uniform";       metric=metric)
    xs_grads, ys_grads = median_scores_for_group(problem, "uniform-grads"; metric=metric)
    (isnothing(xs_std) || isnothing(xs_grads)) && return nothing
    (any(x -> x <= 0, ys_std) || any(x -> x <= 0, ys_grads)) && return nothing

    # Detect linear phases for each curve
    _, _, _, start_std,   end_std,   _ = compute_convergence_slope(xs_std,   ys_std)
    _, _, _, start_grads, end_grads, _ = compute_convergence_slope(xs_grads, ys_grads)

    # Fit log-log regression lines to each curve's linear phase
    slope_s, intercept_s = fit_loglog_line(xs_std,   ys_std,   start_std,   end_std)
    slope_g, intercept_g = fit_loglog_line(xs_grads, ys_grads, start_grads, end_grads)
    (isnothing(slope_s) || isnothing(slope_g)) && return nothing

    # Evaluate both fitted lines at the end of the grads linear phase.
    # At this x, grads has fully converged; standard may still be learning but is
    # within its linear phase (or we extrapolate its fitted line if not).
    x_ref = log10(end_grads)
    log_error_std_pred   = slope_s * x_ref + intercept_s
    log_error_grads_pred = slope_g * x_ref + intercept_g

    return log_error_std_pred - log_error_grads_pred
end

function compute_logdiff_gap_for_problem(problem::AbstractProblem; metric::Symbol=:tv)
    xs_std,   ys_std   = median_scores_for_group(problem, "uniform";       metric=metric)
    xs_grads, ys_grads = median_scores_for_group(problem, "uniform-grads"; metric=metric)
    (isnothing(xs_std) || isnothing(xs_grads)) && return nothing
    (any(x -> x <= 0, ys_std) || any(x -> x <= 0, ys_grads)) && return nothing

    # Use the same overlap logic as compute_constant_gap in plots.jl
    _, _, _, start_std,   end_std,   _ = compute_convergence_slope(xs_std,   ys_std)
    _, _, _, start_grads, end_grads, _ = compute_convergence_slope(xs_grads, ys_grads)
    overlap_start = max(start_std, start_grads)
    overlap_end   = min(end_std,   end_grads)
    overlap_start >= overlap_end && return nothing

    # Both xs arrays start at iteration 3, so index i = iteration i+2 for both.
    # Use xs_std as reference (same convention as compute_constant_gap).
    idxs = findall(x -> overlap_start <= x <= overlap_end, xs_std)
    isempty(idxs) && return nothing
    idxs = filter(i -> i <= length(ys_grads), idxs)
    isempty(idxs) && return nothing

    log_diffs = log10.(ys_std[idxs]) .- log10.(ys_grads[idxs])
    return median(log_diffs)
end

function plot_gaps_vs_dimension_uniform(; metric::Symbol=:tv, save_plot=false, base_fontsize=20, ν=2.5, r_c=2.0)
    set_theme_fonts!(base_fontsize=base_fontsize)

    families = [
        ("MeanGauss",   :blue,      :circle,    [MeanGauss(; x_dim=d) for d in 1:6],               collect(1:6)),
        ("MultidimAB",  :orange,    :rect,      [MultidimProblem(ABProblem(), s) for s in 1:3],     [2,4,6]),
        ("Square",      :green,     :diamond,   [MultidimProblem(SquareProblem(), d) for d in 1:6], collect(1:6)),
        ("Sine",        :red,       :utriangle, [MultidimProblem(SineProblem(), d) for d in 1:6],   collect(1:6)),
        ("Cubic",       :purple,    :dtriangle, [MultidimProblem(CubicProblem(), d) for d in 1:6],  collect(1:6)),
    ]

    metric_label = metric == :convergence ? "L2 risk (simulator)" : "TV metric (posterior)"

    fig = Figure(; size=(1000, 500))
    ax = Axis(fig[1, 1];
        xlabel = "Dimension d",
        ylabel = "log₁₀(error_std) − log₁₀(error_grads)",
        title  = "Log gap between standard and gradient-augmented approaches\n($metric_label, uniform acquisition)",
    )

    all_dims = Int[]
    for (label, color, marker, problems, dims) in families
        gaps = [compute_logdiff_gap_for_problem(p; metric) for p in problems]
        valid = [(d, g) for (d, g) in zip(dims, gaps) if !isnothing(g)]
        isempty(valid) && continue
        append!(all_dims, first.(valid))
        scatterlines!(ax, first.(valid), last.(valid);
            label=label, markersize=12, color=color, marker=marker,
            linewidth=2)
    end

    if !isempty(all_dims)
        d_theory = range(minimum(all_dims) - 0.3, maximum(all_dims) + 0.3; length=200)
        # Theory: dominant constant gap = γ · log₁₀((1+d)/r_c), γ = ν/(2ν+d)
        theory_gap = @. (ν / (2ν + d_theory)) * log10((1 + d_theory) / r_c)
        lines!(ax, d_theory, theory_gap;
            label="Theory: γ·log₁₀((1+d)/r_c), ν=$(ν), r_c=$(r_c)",
            linestyle=:dash, linewidth=2, color=:black)
    end

    Legend(fig[1, 2], ax)

    if save_plot
        mkpath(plot_dir())
        save(plot_dir() * "/gaps_vs_dimension_uniform_$(metric).png", fig)
        save(plot_dir() * "/gaps_vs_dimension_uniform_$(metric).pdf", fig)
        @info "Saved to $(plot_dir())/gaps_vs_dimension_uniform_$(metric).png"
    end
    return fig
end

plot_slopes_vs_dimension(; metric=:tv,          save_plot=true)
plot_slopes_vs_dimension(; metric=:convergence,  save_plot=true)
plot_gaps_vs_dimension_uniform(; metric=:tv,         save_plot=true)
plot_gaps_vs_dimension_uniform(; metric=:convergence, save_plot=true)
@info "Done."
