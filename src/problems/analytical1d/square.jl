"""
    SquareProblem()

A 1D analytical problem with simulator y = x².
Observation z_obs = 1.0 with Gaussian likelihood yields a symmetric bimodal
posterior (peaks near x = ±1).

Extends naturally to d dimensions via MultidimProblem: the averaged simulator
(x₁² + ... + xd²)/d = 1 concentrates the posterior on a d-sphere of radius √d.
"""
@kwdef struct SquareProblem <: AbstractProblem
    gradients::Bool = false
end

set_gradients(p::SquareProblem, val::Bool) = SquareProblem(val)


module SquareProblemModule

import ..SquareProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

# --- API ---

simulator(p::SquareProblem) = p.gradients ? simulation_with_grads : simulation
domain(::SquareProblem)     = Domain(; bounds = ([-3.], [3.]))
likelihood(::SquareProblem) = NormalLikelihood(; z_obs = [1.], std_obs = [0.2])
prior_mean(::SquareProblem) = [0.]
x_prior(::SquareProblem)    = Product([Uniform(-3., 3.)])
est_amplitude(::SquareProblem) = [5.]
est_noise_std(::SquareProblem)      = nothing
est_grad_noise_std(::SquareProblem) = nothing
true_f(::SquareProblem) = simulation

# --- Simulator ---

function simulation(x)
    return [x[1]^2]
end

function simulation_with_grads(x)
    y = [x[1]^2]
    J = reshape([2. * x[1]], 1, 1)
    return y, J
end

end # module SquareProblemModule
