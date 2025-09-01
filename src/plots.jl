include("main.jl")

using ProgressMeter

plot_results(problem::AbstractProblem; kwargs...) = plot_results([problem]; kwargs...)

function plot_results(problems::AbstractVector; save_plot=false, xscale=identity, yscale=log)
    ### setings
    plotted_groups = ["standard", "loglike", "loglike-imiqr", "eiv", "eiig", "nongp"] # TODO
    ###

    # TODO
    scores_by_group = load_stored_scores!(problems)
    # scores_by_group = recalculate_scores!(problems; plotted_groups)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Iteration", ylabel="Median Score", title="Median Scores by Group", xscale, yscale)

    colors = Makie.wong_colors()  # or use any preferred color palette
    group_names = collect(keys(scores_by_group))
    color_map = Dict(group => colors[mod1(i, length(colors))] for (i, group) in enumerate(group_names))

    for (group, scores) in scores_by_group
        group in plotted_groups || continue

        color = color_map[group]
        # scores is a Vector of score histories (each is a Vector)
        # Pad with `missing` to equal length if needed
        maxlen = maximum(length.(scores))
        if !allequal(length.(scores))
            maxlen = maximum(length.(scores))
            successful = sum(length.(scores) .== maxlen)
            @warn "Scores for group \"$group\" have different lengths! Padding with `missing`.
            ($successful/$(length(scores)) runs have the max length of $maxlen)"
        end
        padded = [vcat(s, fill(missing, maxlen - length(s))) for s in scores]
        arr = reduce(hcat, padded)
        xs = 0:maxlen-1 |> collect

        # avoid zero if xscale if logarithmic
        if isinf(xscale(0)) && (xs[1] == 0)
            xs = xs[2:end]
            arr = arr[2:end, :]
        end

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

    save_plot && save(plot_dir() * "/" * plot_name(problems) * ".png", fig)
    fig
end

plot_name(problems::AbstractVector) = join(plot_name.(problems), "_")
plot_name(problem::AbstractProblem) = string(typeof(problem))

function load_stored_scores!(problems::AbstractVector; kwargs...)
    return merge(load_stored_scores!.(problems; kwargs...)...)
end
function load_stored_scores!(problem::AbstractProblem)
    # TODO dir
    dir = data_dir(problem)
    # dir = "data/archive/data_01/" * string(typeof(problems[1]))
    files = sort(Glob.glob(joinpath(dir, "*.jld2")))
    
    scores_by_group = Dict{String, Vector{Vector{Float64}}}()

    for file in files
        fname = split(basename(file), ".")[1]
        group = split(fname, "_")[1]
        
        # startswith(fname, "start") && continue  # skip start files
        endswith(fname, "metric") || continue     # only consider the metric files

        data = load(file)
        @assert haskey(data, "score")
        if !haskey(scores_by_group, group)
            scores_by_group[group] = Vector{Vector{Float64}}()
        end
        push!(scores_by_group[group], data["score"])
    end

    return scores_by_group
end

function recalculate_scores!(problems::AbstractVector; kwargs...)
    return merge(recalculate_scores!.(problems; kwargs...)...)
end
function recalculate_scores!(problem::AbstractProblem; plotted_groups=nothing)
    dir = data_dir(problem)
    files = sort(Glob.glob(joinpath(dir, "*.jld2")))

    scores_by_group = Dict{String, Vector{Vector{Float64}}}()

    # TODO
    xs = rand(x_prior(problem), 20 * 10^x_dim(problem))
    ws = exp.( (0.) .- logpdf.(Ref(x_prior(problem)), eachcol(xs)) )

    # metric = MMDMetric(;
    #     kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
    # )
    # metric = OptMMDMetric(;
    #     kernel = GaussianKernel(),
    #     bounds,
    #     algorithm = BOBYQA(),
    # )
    metric = TVMetric(;
        grid = xs,
        ws = ws,
    )

    # TODO
    sampler = AMISSampler(;
            iters = 10,
            proposal_fitter = BOLFI.AnalyticalFitter(), # re-fit the proposal analytically
            # proposal_fitter = OptimizationFitter(;      # re-fit the proposal by MAP optimization
            #     algorithm = NEWUOA(),
            #     multistart = 24,
            #     parallel = parallel(),
            #     static_schedule = true, # issues with PRIMA.jl
            #     rhoend = 1e-2,
            # ),
            # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
            gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
                algorithm = BOBYQA(),
                multistart = 24,
                parallel = parallel(),
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
        
        # only calculate for the groups from the list
        if !(group in plotted_groups)
            scores_by_group[group] = Vector{Float64}[] # just to keep plot colors consistent
            continue
        end
        
        # startswith(fname, "start") && continue  # skip start files
        endswith(fname, "iters") || continue  # only consider the iters files

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
    sample_count = 2 * 10^x_dim(problem)

    ref = reference(problem)
    if ref isa Function
        true_samples = BOLFI.pure_sample_posterior(sampler, ref, p.problem.domain, sample_count)
    else
        true_samples = ref
    end

    est_logpost = log_posterior_estimate()(p)
    approx_samples = BOLFI.pure_sample_posterior(sampler, est_logpost, p.problem.domain, sample_count)

    score = calculate_metric(metric, true_samples, approx_samples)
    return score
end
function calculate_score(metric::PDFMetric, problem::AbstractProblem, p::BolfiProblem, sampler::DistributionSampler)
    ### retrieve the true and approx logpdf
    ref = reference(problem)
    @assert ref isa Function
    
    true_logpdf = ref
    approx_logpdf = log_posterior_estimate()(p)

    ### calculate metric
    score = calculate_metric(metric, true_logpdf, approx_logpdf)
    return score
end
