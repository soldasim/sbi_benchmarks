"""
    MeanGauss()

A simple problem where the simulator computes the mean of the input vector `x`.
The observation is z_obs = [0] and the likelihood is Gaussian.
"""
@kwdef struct MeanGauss <: AbstractProblem
    gradients::Bool = false
    x_dim::Int = 5
end

set_gradients(p::MeanGauss, val::Bool) = MeanGauss(val, p.x_dim)


module MeanGaussModule

import ..MeanGauss

import ..simulator
import ..domain
import ..y_max
import ..likelihood
import ..prior_mean
import ..x_prior
import ..est_amplitude
import ..est_noise_std
import ..est_grad_noise_std
import ..true_f
import ..reference_samples

using BOSS
using BOSIP
using Distributions
using LinearAlgebra

# --- API ---

simulator(p::MeanGauss) = p.gradients ? mean_simulation_with_grads : mean_simulation

domain(p::MeanGauss) = Domain(;
    bounds = get_bounds(p.x_dim),
)

likelihood(::MeanGauss) = NormalLikelihood(; z_obs = [0.], std_obs = [1.])

prior_mean(::MeanGauss) = [0.]

x_prior(p::MeanGauss) = Product(fill(Normal(0., 1.), p.x_dim))

est_amplitude(::MeanGauss) = [1.]

# TODO noise
est_noise_std(::MeanGauss) = nothing
est_grad_noise_std(::MeanGauss) = nothing

true_f(::MeanGauss) = mean_simulation


# --- UTILS ---

function get_bounds(dim::Int)
    return (fill(-3., dim), fill(3., dim))
end

function mean_simulation(x)
    y = [mean(x)]
    return y
end

function mean_simulation_with_grads(x)
    y = [mean(x)]
    # Jacobian: each component contributes 1/dim to the mean
    J = reshape(fill(1. / length(x), length(x)), 1, length(x))
    return y, J
end

end # module MeanGaussModule
