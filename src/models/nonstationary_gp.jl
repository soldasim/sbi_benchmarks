
"""
    NonstationaryGPModel()

Construct the `NonstationaryGP` surrogate model.
"""
struct NonstationaryGPModel <: AbstractModel end

function construct_model(::NonstationaryGPModel, problem::AbstractProblem)
    x_dim_ = x_dim(problem)
    y_dim_ = y_dim(problem)

    bounds = domain(problem).bounds

    return NonstationaryGP(;
        mean = prior_mean(problem),
        lengthscale_model = BOSS.default_lengthscale_model(bounds, y_dim_),
        amplitude_model = calc_inverse_gamma.(y_extrema(problem)...),
        noise_std_model = noise_std_priors(problem),
    )
end
