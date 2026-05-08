"""
    CubicProblem()

A 1D analytical problem with simulator y = x³ - x = x(x-1)(x+1).
Observation z_obs = 0.0 with tight Gaussian likelihood yields a trimodal
posterior (roots at x = -1, 0, 1); the peak at x = 0 is wider since
|dy/dx| = 1 there vs |dy/dx| = 2 at x = ±1.

Extends naturally to d dimensions via MultidimProblem.
"""
@kwdef struct CubicProblem <: AbstractProblem
    gradients::Bool = false
end

set_gradients(p::CubicProblem, val::Bool) = CubicProblem(val)


module CubicProblemModule

import ..CubicProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

# --- API ---

simulator(p::CubicProblem) = p.gradients ? simulation_with_grads : simulation
domain(::CubicProblem)     = Domain(; bounds = ([-2.], [2.]))
likelihood(::CubicProblem) = NormalLikelihood(; z_obs = [0.], std_obs = [0.1])
prior_mean(::CubicProblem) = [0.]
x_prior(::CubicProblem)    = Product([Uniform(-2., 2.)])
est_amplitude(::CubicProblem) = [3.]
est_noise_std(::CubicProblem)      = nothing
est_grad_noise_std(::CubicProblem) = nothing
true_f(::CubicProblem) = simulation

# --- Simulator ---

function simulation(x)
    return [x[1]^3 - x[1]]
end

function simulation_with_grads(x)
    y = [x[1]^3 - x[1]]
    J = reshape([3. * x[1]^2 - 1.], 1, 1)
    return y, J
end

end # module CubicProblemModule
