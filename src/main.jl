using BOSS
using BOLFI
using Distributions
using KernelFunctions
using LinearAlgebra
using OptimizationPRIMA

using JLD2
using Glob
using CairoMakie

include("include_code.jl")
include("data_paths.jl")
include("generate_starts.jl")
include("plots.jl")

function main(; run_name="_test", save_data=false, data=nothing, run_idx=nothing)
    ### problem
    def_problem = ABProblem()
    # def_problem = SIRProblem()

    ### settings
    init_data_count = 3

    def_model = GaussianProcessModel()
    
    acquisition = PostVarAcq()
    # acquisition = InfoGainInt(;
    #     x_samples = 1000,
    #     samples = 20,
    #     x_proposal = x_prior(def_problem),
    #     y_kernel = BOSS.GaussianKernel(),
    #     p_kernel = BOSS.GaussianKernel(),
    # )
    
    ### utils
    bounds = domain(def_problem).bounds

    ### init data
    data = isnothing(data) ? get_init_data(def_problem, init_data_count) : data
    @info "Initial data:"
    for x in eachcol(data.X)
        println("  $x")
    end

    ### bolfi problem
    problem = construct_bolfi_problem(;
        problem = def_problem,
        data,
        acquisition,
        model = def_model,
    )

    ### algorithms
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 24,
        parallel = true,
        static_schedule = true, # issues with PRIMA.jl
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 24,
        parallel = true,
        static_schedule = true, # issues with PRIMA.jl
        rhoend = 1e-4,
    )

    ### termination condition
    term_cond = IterLimit(10) # TODO

    ### the metric
    metric_cb = MetricCallback(;
        reference = reference(def_problem),
        
        # sampler = RejectionSampler(;
        #     likelihood_maximizer = LikelihoodMaximizer(;
        #         algorithm = BOBYQA(),
        #         multistart = 24,
        #         parallel = true,
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
            #     parallel = true,
            #     static_schedule = true, # issues with PRIMA.jl
            #     rhoend = 1e-2,
            # ),
            # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
            gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
                algorithm = BOBYQA(),
                multistart = 24,
                parallel = true,
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

        # ### plot callback
        # plot_callback = PlotModule.PlotCB(;
        #     problem = def_problem,
        #     plot_each = 1,
        # ),
    )

    options = BolfiOptions(;
        callback = metric_cb,
    )

    ### RUN
    bolfi!(problem; model_fitter, acq_maximizer, term_cond, options)

    ### save results
    dir = data_dir(def_problem)
    filename = data_filename(def_problem, run_name, run_idx)
    
    if save_data
        mkpath(dir)
        save(filename * "_extras.jld2", Dict(
            "run_idx" => run_idx,
            "problem" => problem,
            "model_fitter" => model_fitter,
            "acq_maximizer" => acq_maximizer,
            "term_cond" => term_cond,
            "options" => options,
            "metric" => metric_cb,
        ))
        save(filename * ".jld2", Dict(
            "run_idx" => run_idx,
            "data" => (problem.problem.data.X, problem.problem.data.Y),
            "score" => metric_cb.score_history,
        ))
    end

    return problem
end

function run(problem::AbstractProblem)
    run_name = "ig" # the name used for storing data from this run

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
