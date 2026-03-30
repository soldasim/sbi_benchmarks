
"""
    GaussProblem()

The "simple" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

In contrast to the problem as defined in the paper, here the input vector `x`
is returned as the simulator output (i.e. the simulator is just the identity function).
See the `LogGaussProblem` for the original version of the problem.
"""
@kwdef struct GaussProblem <: AbstractProblem
    gradients::Bool = false
    x_dim::Int = 5
end

set_gradients(p::GaussProblem, val::Bool) = GaussProblem(val, p.x_dim)


module GaussProblemModule

import ..GaussProblem

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
using Bijectors
using LinearAlgebra

# --- API ---

simulator(p::GaussProblem) = p.gradients ? simulation_with_grads : simulation

domain(p::GaussProblem) = Domain(;
    bounds = get_bounds(p.x_dim),
)

likelihood(p::GaussProblem) = get_likelihood(p.x_dim)

prior_mean(p::GaussProblem) = zeros(p.x_dim)

x_prior(p::GaussProblem) = get_x_prior(p.x_dim)

est_amplitude(p::GaussProblem) = fill(1., p.x_dim)

# TODO noise
est_noise_std(::GaussProblem) = nothing
est_grad_noise_std(::GaussProblem) = nothing

true_f(::GaussProblem) = simulation


# - - - PARAMETER DOMAIN - - - - -

get_bounds(dim::Int) = (fill(-1., dim), fill(1., dim))


# - - - EXPERIMENT - - - - -

const σ = 0.020
const ρ = 0.005

function get_covariance_matrix(dim::Int)
    # Create a dim x dim matrix with σ on the diagonal and ρ on the off-diagonal
    Σ = fill(ρ, dim, dim)
    for i in 1:dim
        Σ[i, i] = σ
    end
    return Σ
end

# f_(x) = -(1/2) * x' * inv_S * x

function simulation(x)
    return x
end
function simulation_with_grads(x)
    y = x
    J = Matrix(1.0 * I, length(x), length(x))
    return y, J
end

function get_likelihood(dim::Int)
    Σ = get_covariance_matrix(dim)
    return MvNormalLikelihood(;
        z_obs = zeros(dim),
        Σ_obs = Σ,
    )
end

# truncate the prior to the bounds
function get_x_prior(dim::Int)
    return Product(Uniform.(get_bounds(dim)...))
end

end # module GaussProblemModule
