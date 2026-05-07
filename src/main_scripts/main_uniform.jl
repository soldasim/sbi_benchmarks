### Uniform random acquisition (no acquisition function optimization).
### For empirically testing Teckentrup convergence bounds.
### Uses RandomAM() so next point is drawn uniformly from the domain.

using BOSS
using BOSIP
using Distributions
using KernelFunctions
using LinearAlgebra
using OptimizationPRIMA
using Bijectors

using JLD2
using Glob
using CairoMakie
using ProgressMeter

using Random
Random.seed!(555)

parallel() = false

include(pwd() * "/src/include_code.jl")

### START A NEW RUN ###
function main(problem::AbstractProblem; data=nothing, iters=100, kwargs...)
    init_data_count = 3

    if isnothing(data)
        data = get_init_data(problem, init_data_count)
    else
        @assert data isa AbstractMatrix{<:Real}
        sim = simulator(problem)
        X = data
        Y = reduce(hcat, (sim(x) for x in eachcol(X)))[:,:]
        data = BOSS.ExperimentData(X, Y)
    end

    @info "Initial data:"
    for (x, y) in zip(eachcol(data.X), eachcol(data.Y))
        println("  $x -> $y")
    end

    estimator = log_posterior_mean

    model = GaussianProcess(;
        mean = prior_mean(problem),
        kernel = BOSS.Matern52Kernel(),
        lengthscale_priors = get_lengthscale_priors(problem),
        amplitude_priors = get_amplitude_priors(problem),
        noise_std_priors = get_noise_std_priors(problem),
    )

    # Acquisition function is ignored by RandomAM but still required by the interface.
    acquisition = MaxVar()

    bosip = construct_bosip_problem(;
        problem,
        data,
        acquisition,
        model,
    )

    data_max = size(data.X, 2) + iters

    return main(problem, bosip, estimator; data_max, kwargs...)
end

### CONTINUE A RUN ###
function main_continue(problem::AbstractProblem, run_name::String, run_idx::Union{Nothing, Int}; iters=200, kwargs...)
    if isnothing(run_idx)
        file = joinpath(data_dir(problem), "$(run_name)_problem.jld2")
    else
        file = joinpath(data_dir(problem), "$(run_name)_$(run_idx)_problem.jld2")
    end
    bosip = load(file)["problem"]
    @assert bosip isa BosipProblem

    estimator = log_posterior_mean
    @warn "using posterior estimator: $(estimator |> nameof |> string)"

    data_count = size(bosip.problem.data.X, 2)
    @assert data_count >= 3 + 100
    data_max = 3 + iters

    return main(problem, bosip, estimator; continued=true, run_name, run_idx, data_max, kwargs...)
end

function main(problem::AbstractProblem, bosip::BosipProblem, estimator::Function;
    run_name = "test",
    save_data = false,
    metric = false,
    convergence = false,
    plots = false,
    run_idx = nothing,
    continued = false,
    data_max = 1,
    kwargs...
)
    bounds = bosip.problem.domain.bounds

    ### ALGORITHMS ###
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 2,
        warm_start = true,
        parallel = parallel(),
        rhoend = 1e-4,
    )
    # Draw next point uniformly at random from the domain, ignoring acquisition function.
    acq_maximizer = RandomAM()

    term_cond = DataLimit(data_max)

    sampler = AMISSampler(;
        iters = 10,
        proposal_fitter = BOSIP.AnalyticalFitter(),
        gauss_mix_options = GaussMixOptions(;
            algorithm = BOBYQA(),
            multistart = 24,
            parallel = parallel(),
            cluster_ϵs = nothing,
            rel_min_weight = 1e-8,
            rhoend = 1e-4,
        ),
    )

    if metric
        grid_data = load_grid(problem)
        metric_ = TVMetric(;
            grid = grid_data.xs,
            log_ws = grid_data.log_ws,
            true_logvals = grid_data.true_logvals,
        )

        if metric_ isa PDFMetric
            ref = true_logpost(problem)
        else
            ref = reference_samples(problem)
            isnothing(ref) && (ref = true_logpost(problem))
        end
        @assert !isnothing(ref)

        if continued
            metric_cb = reload_metric_cb(metric_, problem, run_name, run_idx)
        else
            metric_cb = MetricCallback(;
                reference = ref,
                logpost_estimator = estimator,
                sampler,
                sample_count = 2 * 10^x_dim(problem),
                metric = metric_,
            )
        end
    end

    if convergence
        conv_data = load_simulator_grid(problem)
        conv_cb = ConvergenceCallback(;
            convergence_metric = l2_norm,
            xs = conv_data.xs,
            log_ws = conv_data.log_ws,
            true_sim_outputs = conv_data.sim_outputs,
        )
    end

    callbacks = BosipCallback[]
    metric && push!(callbacks, metric_cb)
    convergence && push!(callbacks, conv_cb)

    plot_cb = PlotModule.PlotCB(;
        problem,
        estimator,
        sampler,
        sample_count = 2 * 10^x_dim(problem),
        resolution = 200,
        plot_each = 10,
        save_plots = true,
    )
    plots && push!(callbacks, plot_cb)

    data_cb = SaveCallback(;
        dir = data_dir(problem),
        filename = base_filename(problem, run_name, run_idx),
        continued,
    )
    save_data && push!(callbacks, data_cb)

    options = BosipOptions(;
        callback = BOSIP.CombinedCallback(callbacks...),
        info = true,
        debug = false,
    )

    bosip!(bosip; model_fitter, acq_maximizer, term_cond, options)
    return bosip
end

function reload_metric_cb(metric::DistributionMetric, problem::AbstractProblem, run_name::String, run_idx::Union{Nothing, Int})
    dir = data_dir(problem)
    if isnothing(run_idx)
        metric_file = dir * "/$(run_name)_$(metric_fname(Base.typename(typeof(metric)).wrapper)).jld2"
    else
        metric_file = dir * "/$(run_name)_$(run_idx)_$(metric_fname(Base.typename(typeof(metric)).wrapper)).jld2"
    end

    metric_data = load(metric_file)
    score_ = metric_data["score"]
    metric_cb_ = metric_data["metric"]

    @assert typeof(metric_cb_.metric) == typeof(metric)
    return metric_cb_
end
