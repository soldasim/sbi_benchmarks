using JLD2
using Glob
using CairoMakie
using ProgressMeter
using BOSS, BOSIP
using Distributions

using Random
Random.seed!(555)

include(pwd() * "/src/include_code.jl")

### START A NEW RUN ###
function main(problem::AbstractProblem; data=nothing, iters=100, kwargs...)
    ### SETTINGS ###
    init_data_count = 3 # TODO

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

    # ### domain mean as only initial point
    # X = hcat(mean(domain(problem).bounds))
    # sim = simulator(problem)
    # Y = reduce(hcat, (sim(x) for x in eachcol(X)))[:,:]
    # data = ExperimentData(X, Y)

    @info "Initial data:"
    for (x, y) in zip(eachcol(data.X), eachcol(data.Y))
        println("  $x -> $y")
    end


    data_max = size(data.X, 2) + iters

    return run_simformer(problem; data_max, kwargs...)
end

### CONTINUE A RUN ###
function main_continue(problem::AbstractProblem, run_name::String, run_idx::Union{Nothing, Int}; iters=200, kwargs...)
    @assert false # not implemented for Simformer
    # run_simformer(...)
end

function run_simformer(problem::AbstractProblem;
    run_name = "test",
    save_data = false,
    metric = false,
    plots = false,
    run_idx = nothing,
    data_max = 1,
)
    ### INITIALIZE ###
    @pyexec """
    import random
    import torch
    import jax
    import numpy as np
    from hydra import initialize, compose
    from omegaconf import DictConfig
    from scoresbibm.tasks import get_task
    from scoresbibm.methods.method_base import get_method
    from scoresbibm.evaluation import get_metric, eval_inference_task
    from scoresbibm.tasks.base_task import InferenceTask

    ### UTILS
    def set_seed(seed:int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        with jax.default_device(jax.devices("cpu")[0]):
            key = jax.random.PRNGKey(seed)
        return key
    
    ### SETTINGS
    rng = set_seed(555)
    method_name = "score_transformer"

    ### MAIN FUNCTION
    # TODO

    """
    
    ### PERFORMANCE METRIC ###
    # The initialization of the metric may take some time.
    # (in case the reference is being pre-calculated)
    if metric
        # metric_ = MMDMetric(;
        #     kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
        # )
        # metric_ = OptMMDMetric(;
        #     kernel = GaussianKernel(),
        #     bounds,
        #     algorithm = BOBYQA(),
        #     rhoend = 1e-4,
        # )
        grid_data = load_grid(problem)
        metric_ = TVMetric(;
            grid = grid_data.xs,
            log_ws = grid_data.log_ws,
            true_logvals = grid_data.true_logvals,
        )

        # Get a reference appropriate for the used metric is available.
        if metric_ isa PDFMetric
            ref = true_logpost(problem)
        else
            ref = reference_samples(problem)
            isnothing(ref) && (ref = true_logpost(problem))
        end
        @assert !isnothing(ref)

        # Evaluate the metric
        # TODO
    end

    ### PLOTS ###
    if plots
        # TODO
    end
    
    ### STORING RESULTS ###
    # TODO
    # data_cb = SaveCallback(;
    #     dir = data_dir(problem),
    #     filename = base_filename(problem, run_name, run_idx),
    #     continued,
    # )
    # save_data && push!(callbacks, data_cb)

    # return ... # TODO
    return nothing
end
