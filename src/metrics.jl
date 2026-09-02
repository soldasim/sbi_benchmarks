
metric_fname(::Type{OptMMDMetric}) = "metric"  # I am too lazy to rename the files.
metric_fname(::Type{TVMetric}) = "TVmetric"

function get_metric(::Type{MMDMetric}, problem::AbstractProblem)
    # TODO
    bounds = domain(problem).bounds
    λs = (bounds[2] .- bounds[1]) ./ 3

    return MMDMetric(;
        kernel = with_lengthscale(GaussianKernel(), λs),
    )
end
function get_metric(::Type{OptMMDMetric}, problem::AbstractProblem)
    return OptMMDMetric(;
        kernel = GaussianKernel(),
        bounds,
        algorithm = BOBYQA(),
    )
end
function get_metric(::Type{TVMetric}, problem::AbstractProblem)
    if isfile(posterior_grid_filepath(problem))
        grid_data = load_grid(problem)
        return TVMetric(;
            grid = grid_data.xs,
            log_ws = grid_data.log_ws,
            true_logvals = grid_data.true_logvals,
        )
    else
        xs = rand(x_prior(problem), 20 * 10^x_dim(problem))
        log_ws = 0. .- logpdf.(Ref(x_prior(problem)), eachcol(xs))
        return TVMetric(;
            grid = xs,
            log_ws = log_ws,
            true_logpost = true_logpost(problem),
        )
    end
end
