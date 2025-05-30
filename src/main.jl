using BOSS
using BOLFI
using Distributions
using KernelFunctions

using OptimizationPRIMA

include("include.jl")

function main()
    ### settings
    init_data_count = 3

    def_problem = ABProblem()
    # def_problem = SIRProblem()

    def_model = GaussianProcessModel()
    
    acquisition = PostVarAcq()
    
    ### utils
    bounds = domain(def_problem).bounds

    ### bolfi problem
    problem = construct_bolfi_problem(;
        problem = def_problem,
        data = get_init_data(def_problem, init_data_count),
        acquisition,
        model = def_model,
    )

    ### algorithms
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 200,
        parallel = false, # issues with PRIMA.jl
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 200,
        parallel = false, # issues with PRIMA.jl
        rhoend = 1e-4,
    )

    ### termination condition
    term_cond = IterLimit(10)

    ### the metric
    metric_cb = MetricCallback(;
        reference = reference(def_problem),
        
        # sampler = RejectionSampler(;
        #     likelihood_maximizer = LikelihoodMaximizer(;
        #         algorithm = BOBYQA(),
        #         multistart = 200,
        #         parallel = false, # issues with PRIMA.jl
        #         rhoend = 1e-4,
        #     ),
        # ),
        sampler = AMISSampler(;
            iters = 10,
            proposal_fitter = BOLFI.AnalyticalFitter(), # re-fit the proposal analytically
            # proposal_fitter = OptimizationFitter(;      # re-fit the proposal by MAP optimization
            #     algorithm = NEWUOA(),
            #     multistart = 24,
            #     parallel = false, # issues with PRIMA.jl
            #     rhoend = 1e-2,
            # ),
            # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
            gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
                algorithm = BOBYQA(),
                multistart = 200,
                parallel = false, # issues with PRIMA.jl
                cluster_ϵs = nothing,
                rel_min_weight = 1e-8,
                rhoend = 1e-4,
            ),
        ),
        
        sample_count = 1200,
        metric = MMDMetric(;
            kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
        ),
    )

    options = BolfiOptions(;
        callback = metric_cb,
    )

    ### RUN
    bolfi!(problem; model_fitter, acq_maximizer, term_cond, options)
end
