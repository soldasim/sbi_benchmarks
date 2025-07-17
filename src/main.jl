using BOSS
using BOLFI
using Distributions
using KernelFunctions
using LinearAlgebra
using OptimizationPRIMA

using JLD2
using Glob
using CairoMakie

# TODO
# posterior_estimate() = approx_posterior
posterior_estimate() = posterior_mean

include("include_code.jl")
include("data_paths.jl")
include("generate_starts.jl")

function main(; run_name="_test", save_data=false, data=nothing, run_idx=nothing)
    ### PROBLEM ###
    problem = ABProblem()
    # problem = SIRProblem()
    # problem = SimpleProblem()

    
    ### SETTINGS ###
    init_data_count = 3
    parallel = true # PRIMA.jl causes StackOverflow when parallelized on Linux

    
    ### SURROGATE MODEL ###
    x_dim_ = x_dim(problem)
    y_dim_ = y_dim(problem)
    bounds = domain(problem).bounds
    d = (bounds[2] .- bounds[1])

    # # TODO SimpleProblem custom model
    # model = SimpleProblemModule.get_model()

    model = GaussianProcess(;
        mean = prior_mean(problem),
        kernel = BOSS.Matern32Kernel(),
        lengthscale_priors = fill(product_distribution(calc_inverse_gamma.(d ./ 20, d)), y_dim_),
        amplitude_priors = calc_inverse_gamma.(y_extrema(problem)...),
        noise_std_priors = noise_std_priors(problem),
    )
    # model = NonstationaryGP(;
    #     mean = prior_mean(problem),
    #     lengthscale_model = BOSS.default_lengthscale_model(bounds, y_dim_),
    #     amplitude_model = calc_inverse_gamma.(y_extrema(problem)...),
    #     noise_std_model = noise_std_priors(problem),
    # )
    
    
    ### ACQUISITION ###
    acquisition = PostVarAcq()
    # acquisition = LogPostVarAcq()
    # acquisition = InfoGainInt(;
    #     x_samples = 1000,
    #     samples = 20,
    #     x_proposal = x_prior(problem),
    #     y_kernel = BOSS.GaussianKernel(),
    #     p_kernel = BOSS.GaussianKernel(),
    # )

    
    ### INIT DATA ###
    data = isnothing(data) ? get_init_data(problem, init_data_count) : data
    @info "Initial data:"
    for x in eachcol(data.X)
        println("  $x")
    end

    
    ### BOLFI PROBLEM ###
    bolfi = construct_bolfi_problem(;
        problem,
        data,
        acquisition,
        model,
    )

    
    ### ALGORITHMS ###
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 24,
        parallel,
        static_schedule = true, # issues with PRIMA.jl
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 24,
        parallel,
        static_schedule = true, # issues with PRIMA.jl
        rhoend = 1e-4,
    )

    
    ### TERMINATION CONDITION ###
    term_cond = IterLimit(50) # TODO

    
    ### PERFORMANCE METRIC ###
    metric_cb = MetricCallback(;
        reference = reference(problem),
        
        # sampler = RejectionSampler(;
        #     likelihood_maximizer = LikelihoodMaximizer(;
        #         algorithm = BOBYQA(),
        #         multistart = 24,
        #         parallel,
        #         static_schedule = true, # issues with PRIMA.jl
        #         rhoend = 1e-4,
        #     ),
        # ),
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
        ),
        
        sample_count = 1000,
        metric = MMDMetric(;
            kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
        ),

        # TODO
        # ### plot callback
        # plot_callback = PlotModule.PlotCB(;
        #     problem,
        #     plot_each = 1,
        #     save_plots = true,
        # ),
    )

    
    ### STORING RESULTS ###
    dir = data_dir(problem)
    filepath = data_filepath(problem, run_name, run_idx)

    if save_data
        callback = CombinedCallback(
            metric_cb,
            SaveCallback(; dir, filepath),
        )
    else
        callback = CombinedCallback(
            metric_cb,
        )
    end

    options = BolfiOptions(;
        callback,
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
    metric_cb = options.callback.callback.callbacks[1]
    @assert metric_cb isa MetricCallback

    mkpath(cb.dir)

    save(cb.filepath * "_extras.jld2", Dict(
        "run_idx" => cb.run_idx,
        "problem" => problem,
        "model_fitter" => model_fitter,
        "acq_maximizer" => acq_maximizer,
        "term_cond" => term_cond,
        "options" => options,
        "metric" => metric_cb,
    ))
    save(cb.filepath * ".jld2", Dict(
        "run_idx" => cb.run_idx,
        "data" => (problem.problem.data.X, problem.problem.data.Y),
        "score" => metric_cb.score_history,
    ))

    if first
        iters = [problem]
    else
        iters = load(cb.filepath * "_iters.jld2")["problems"]
        push!(iters, problem)
    end
    save(cb.filepath * "_iters.jld2", Dict("problems" => iters))
end
