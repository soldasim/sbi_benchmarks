"""
    SineProblem()

A 1D analytical problem with simulator y = sin(x).
Observation z_obs = 0.7 with tight Gaussian likelihood yields a trimodal
posterior (three solutions to sin(x) ≈ 0.7 lie within the domain [-4, 4]).

Extends naturally to d dimensions via MultidimProblem: the averaged simulator
(sin(x₁) + ... + sin(xd))/d tests multimodal inference in high dimensions,
where the CLT progressively merges modes.
"""
@kwdef struct SineProblem <: AbstractProblem
    gradients::Bool = false
end

set_gradients(p::SineProblem, val::Bool) = SineProblem(val)


module SineProblemModule

import ..SineProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

# --- API ---

simulator(p::SineProblem) = p.gradients ? simulation_with_grads : simulation
domain(::SineProblem)     = Domain(; bounds = ([-4.], [4.]))
likelihood(::SineProblem) = NormalLikelihood(; z_obs = [0.7], std_obs = [0.1])
prior_mean(::SineProblem) = [0.]
x_prior(::SineProblem)    = Product([Uniform(-4., 4.)])
est_amplitude(::SineProblem) = [1.]
est_noise_std(::SineProblem)      = nothing
est_grad_noise_std(::SineProblem) = nothing
true_f(::SineProblem) = simulation

# --- Simulator ---

function simulation(x)
    return [sin(x[1])]
end

function simulation_with_grads(x)
    y = [sin(x[1])]
    J = reshape([cos(x[1])], 1, 1)
    return y, J
end

end # module SineProblemModule
