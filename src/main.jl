using BOSS
using BOLFI
using Distributions
using KernelFunctions
using LinearAlgebra
using OptimizationPRIMA
using Bijectors

using JLD2
using Glob
using CairoMakie

using Random
Random.seed!(555)

# TODO
log_posterior_estimate() = log_posterior_mean
# log_posterior_estimate() = log_approx_posterior

parallel() = false # PRIMA.jl causes StackOverflow when parallelized on Linux

include("include_code.jl")
include("data_paths.jl")
include("generate_starts.jl")
include("param_priors.jl")

### PROBLEM ###
get_problem() = ABProblem()
# get_problem() = SimpleProblem()
# get_problem() = SIRProblem()

function main(; data=nothing, kwargs...)
    problem = get_problem()
    

    ### SETTINGS ###
    init_data_count = 3

    
    ### SURROGATE MODEL ###
    model = GaussianProcess(;
        mean = prior_mean(problem),
        kernel = BOSS.Matern32Kernel(),
        lengthscale_priors = get_lengthscale_priors(problem),
        amplitude_priors = get_amplitude_priors(problem),
        noise_std_priors = get_noise_std_priors(problem),
    )
    # model = NonstationaryGP(;
    #     mean = prior_mean(problem),
    #     lengthscale_model = BOSS.default_lengthscale_model(bounds(problem), y_dim(problem)),
    #     amplitude_model = get_amplitude_priors(problem),
    #     noise_std_model = get_noise_std_priors(problem),
    # )
    
    
    ### ACQUISITION ###
    acquisition = MaxVar()
    # acquisition = LogMaxVar()
    # acquisition = EIIG(;
    #     y_samples = 20,
    #     x_samples = 2 * 10^x_dim(problem),
    #     x_proposal = x_prior(problem),
    #     y_kernel = BOSS.GaussianKernel(),
    #     p_kernel = BOSS.GaussianKernel(),
    # )
    # acquisition = EIV(
    #     y_samples = 20,
    #     x_samples = 2 * 10^x_dim(problem),
    #     x_proposal = x_prior(problem),
    # )
    # acquisition = IMIQR(;
    #     p_u = 0.75,
    #     x_samples = 2 * 10^x_dim(problem),
    #     x_proposal = x_prior(problem),
    # )

    
    ### INIT DATA ###
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

    
    ### BOLFI PROBLEM ###
    bolfi = construct_bolfi_problem(;
        problem,
        data,
        acquisition,
        model,
    )

    return main(bolfi; kwargs...)
end

# for continuing an experiment (mainly for debugging)
function main(bolfi::BolfiProblem; run_name="_test", save_data=false, metric=false, plots=false, run_idx=nothing)
    problem = get_problem()
    bounds = bolfi.problem.domain.bounds


    ### ALGORITHMS ###
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 24,
        parallel = parallel(),
        rhoend = 1e-4,
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 24,
        parallel = parallel(),
        rhoend = 1e-4,
    )

    
    ### TERMINATION CONDITION ###
    term_cond = IterLimit(100) # TODO


    ### SAMPLER ###
    # sampler = RejectionSampler(;
    #     logpdf_maximizer = LogpdfMaximizer(;
    #         algorithm = BOBYQA(),
    #         multistart = 24,
    #         parallel = parallel(),
    #         static_schedule = true, # issues with PRIMA.jl
    #         rhoend = 1e-4,
    #     ),
    # )
    sampler = AMISSampler(;
        iters = 10,
        proposal_fitter = BOLFI.AnalyticalFitter(), # re-fit the proposal analytically
        # proposal_fitter = OptimizationFitter(;      # re-fit the proposal by MAP optimization
        #     algorithm = NEWUOA(),
        #     multistart = 6,
        #     parallel = parallel(),
        #     static_schedule = true, # issues with PRIMA.jl
        #     rhoend = 1e-2,
        # ),
        # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
        gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
            algorithm = BOBYQA(),
            multistart = 24,
            parallel = parallel(),
            cluster_ϵs = nothing,
            rel_min_weight = 1e-8,
            rhoend = 1e-4,
        ),
    )

    
    ### PERFORMANCE METRIC ###
    metric_cb = MetricCallback(;
        reference = reference(problem),
        logpost_estimator = log_posterior_estimate(),
        sampler,
        sample_count = 2 * 10^x_dim(problem),
        metric = MMDMetric(;
            kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
        ),
    )
    # first callback in `callbacks` (this is important for `SaveCallback`)
    callbacks = BolfiCallback[]
    metric && push!(callbacks, metric_cb)


    ### PLOTS ###
    plot_cb = PlotModule.PlotCB(;
        problem,
        sampler,
        sample_count = 2 * 10^x_dim(problem),
        plot_each = 10,
        save_plots = true,
    )
    plots && push!(callbacks, plot_cb)

    
    ### STORING RESULTS ###
    dir = data_dir(problem)
    filepath = data_filepath(problem, run_name, run_idx)
    data_cb = SaveCallback(; dir, filepath)
    save_data && push!(callbacks, data_cb)

    options = BolfiOptions(;
        callback = CombinedCallback(callbacks...),
    )

    
    ### RUN ###
    bolfi!(bolfi; model_fitter, acq_maximizer, term_cond, options)
    return bolfi
end

function run(problem::AbstractProblem)
    run_name = "loglike" # the name used for storing data from this run

    start_files = Glob.glob(data_dir(problem) * "/start_*.jld2")
    @info "Running $(length(start_files)) runs of the $(typeof(problem)) ..."
    
    for start_file in start_files
        m = match(r"start_(\d+)\.jld2$", start_file)
        run_idx = parse(Int, m.captures[1])
        data = load(start_file, "data")
        main(; run_name, save_data=true, data, run_idx)
    end

    nothing
end

"""
    SaveCallback(; kwargs...)

Saves the run data after every iteration (by overwriting the data stored in the previous iteration).

# Keywords
- `dir::String`: Directory to save the data to.
- `filepath::String`: Filepath to save the data to (without extension).
- `run_idx::Union{Int, Nothing} = nothing`
"""
@kwdef struct SaveCallback <: BolfiCallback
    dir::String
    filepath::String
    run_idx::Union{Int, Nothing} = nothing
end

function (cb::SaveCallback)(problem::BolfiProblem; first, model_fitter, acq_maximizer, term_cond, options)
    mkpath(cb.dir)

    # problem & data
    save(cb.filepath * "_problem.jld2", Dict(
        "problem" => problem,
    ))
    save(cb.filepath * "_data.jld2", Dict(
        "run_idx" => cb.run_idx,
        "data" => (problem.problem.data.X, problem.problem.data.Y),
    ))

    # other
    save(cb.filepath * "_extras.jld2", Dict(
        "run_idx" => cb.run_idx,
        "problem" => problem,
        "model_fitter" => model_fitter,
        "acq_maximizer" => acq_maximizer,
        "term_cond" => term_cond,
        "options" => options,
    ))

    # metric
    if !isempty(options.callback.callback.callbacks)
        metric_cb = options.callback.callback.callbacks[1]
        if metric_cb isa MetricCallback
            save(cb.filepath * "_metric.jld2", Dict(
                "score" => metric_cb.score_history,
                "metric" => metric_cb,
            ))
        end
    end

    if first
        iters = [problem]
    else
        iters = load(cb.filepath * "_iters.jld2")["problems"]
        push!(iters, problem)
    end
    save(cb.filepath * "_iters.jld2", Dict("problems" => iters))
end
