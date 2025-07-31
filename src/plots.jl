include("main.jl")

using ProgressMeter

function plot_results(problem::AbstractProblem; save_plot=false)
    ### setings
    dir = data_dir(problem)
    plotted_groups = ["TEST"] # TODO
    ###

    files = sort(Glob.glob(joinpath(dir, "*.jld2")))
    scores_by_group = Dict{String, Vector{Vector{Float64}}}()

    # TODO
    load_stored_scores!(scores_by_group, files, problem)
    # recalculate_scores!(scores_by_group, files, problem)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Iteration", ylabel="Median Score", title="Median Scores by Group", yscale=log)

    colors = Makie.wong_colors()  # or use any preferred color palette
    group_names = collect(keys(scores_by_group))
    color_map = Dict(group => colors[mod1(i, length(colors))] for (i, group) in enumerate(group_names))

    for (group, scores) in scores_by_group
        group in plotted_groups || continue

        color = color_map[group]
        # scores is a Vector of score histories (each is a Vector)
        # Pad with `missing` to equal length if needed
        maxlen = maximum(length.(scores))
        allequal(length.(scores)) || @warn "Scores for group $group have different lengths, padding with `missing`."
        padded = [vcat(s, fill(missing, maxlen - length(s))) for s in scores]
        arr = reduce(hcat, padded)
        xs = 0:maxlen-1

        # Compute statistics ignoring NaNs
        median_scores = mapslices(median∘skipmissing, arr; dims=2)[:]
        q25 = mapslices(x -> quantile(skipmissing(x), 0.25), arr; dims=2)[:]
        q75 = mapslices(x -> quantile(skipmissing(x), 0.75), arr; dims=2)[:]
        # Plot median line
        lines!(ax, xs, median_scores, label=group, color=color)
        # Plot quantile band with alpha
        band!(ax, xs, q25, q75; color=color, alpha=0.6)

        q10 = mapslices(x -> quantile(skipmissing(x), 0.1), arr; dims=2)[:]
        q90 = mapslices(x -> quantile(skipmissing(x), 0.9), arr; dims=2)[:]
        lines!(ax, xs, q10; color=color, linestyle=:dot, linewidth=1)
        lines!(ax, xs, q90; color=color, linestyle=:dot, linewidth=1)
    end

    axislegend(ax)

    save_plot && save(plot_dir() * "/" * string(typeof(problem)) * ".png", fig)
    fig
end

function load_stored_scores!(scores_by_group, files, problem)
    for file in files
        fname = split(basename(file), ".")[1]
        group = split(fname, "_")[1]
        
        startswith(fname, "start") && continue  # skip start files
        endswith(fname, "extras") && continue   # skip extras files
        endswith(fname, "iters") && continue    # skip iters files

        data = load(file)
        @assert haskey(data, "score")
        if !haskey(scores_by_group, group)
            scores_by_group[group] = Vector{Vector{Float64}}()
        end
        push!(scores_by_group[group], data["score"])
    end
end

function recalculate_scores!(scores_by_group, files, problem)
    bounds = domain(problem).bounds

    # TODO
    metric = MMDMetric(;
        kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
    )

    # TODO
    sampler = AMISSampler(;
            iters = 10,
            proposal_fitter = BOLFI.AnalyticalFitter(), # re-fit the proposal analytically
            # proposal_fitter = OptimizationFitter(;      # re-fit the proposal by MAP optimization
            #     algorithm = NEWUOA(),
            #     multistart = 24,
            #     parallel,
            #     static_schedule = true, # issues with PRIMA.jl
            #     rhoend = 1e-2,
            # ),
            # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
            gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
                algorithm = BOBYQA(),
                multistart = 24,
                parallel,
                static_schedule = true, # issues with PRIMA.jl
                cluster_ϵs = nothing,
                rel_min_weight = 1e-8,
                rhoend = 1e-4,
            ),
        )

    for file in files
        fname = split(basename(file), ".")[1]
        group = split(fname, "_")[1]
        run_idx = split(fname, "_")[2]
        
        # startswith(fname, "start") && continue  # skip start files
        endswith(fname, "iters") || continue    # only consider the iters files

        iters = load(file)["problems"]

        if !haskey(scores_by_group, group)
            scores_by_group[group] = Vector{Vector{Float64}}()
        end

        # Recalculate the score using the MetricCallback

        scores = zeros(length(iters))
        @showprogress desc="Calculating run_$run_idx" for (idx, itr) in enumerate(iters)
            scores[idx] = calculate_score(metric, problem, itr, sampler)
        end
        push!(scores_by_group[group], scores)
    end

    return scores_by_group
end

function calculate_score(metric::SampleMetric, problem::AbstractProblem, p::BolfiProblem, sampler::DistributionSampler)
    # TODO
    sample_count = 1000

    ref = reference(problem)
    if ref isa Function
        true_samples = pure_sample_posterior(sampler, ref, p.problem.domain, sample_count)
    else
        true_samples = ref
    end

    est_logpost = log_posterior_estimate()(p)
    approx_samples = pure_sample_posterior(sampler, est_logpost, p.problem.domain, sample_count)

    score = calculate_metric(metric, true_samples, approx_samples)
    return score
end
