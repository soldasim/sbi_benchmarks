
"""
    GaussianProcessModel()

Construct the `GaussianProcess` surrogate model.
"""
struct GaussianProcessModel <: AbstractModel end

function construct_model(::GaussianProcessModel, problem::AbstractProblem)
    x_dim_ = x_dim(problem)
    y_dim_ = y_dim(problem)

    bounds = domain(problem).bounds
    d = (bounds[2] .- bounds[1])
    
    return GaussianProcess(;
        kernel = BOSS.Matern32Kernel(),
        lengthscale_priors = fill(product_distribution(calc_inverse_gamma.(d ./ 20, d)), y_dim_),
        amplitude_priors = calc_inverse_gamma.(y_extrema(problem)...),
        noise_std_priors = noise_std_priors(problem),
    )
end
