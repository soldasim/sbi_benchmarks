"""
    StyblinskiTangProblem(; x_dim=2)

A d-dimensional problem with the Styblinski-Tang simulator.

Simulator: y = ½ Σᵢ₌₁ᵈ (xᵢ⁴ − 16xᵢ² + 5xᵢ)  (scalar output)

The function is separable: each dimension contributes independently. The 1D
component g(t) = ½(t⁴ − 16t² + 5t) has two local minima at t ≈ −2.903
(g ≈ −39.17, global) and t ≈ 2.747 (g ≈ −25.1). This separable structure
means the response surface has 2^d local minima in d dimensions.

With z_obs = 0 and Gaussian likelihood, the posterior is multimodal — the
level set {f(x) ≈ 0} crosses multiple basins — making this a good benchmark
for testing acquisition functions on multimodal posteriors and for illustrating
how a non-injective proxy (e.g. modeling |y| instead of y) loses information.
"""
@kwdef struct StyblinskiTangProblem <: AbstractProblem
    gradients::Bool = false
    x_dim::Int = 2
end

set_gradients(p::StyblinskiTangProblem, val::Bool) = StyblinskiTangProblem(val, p.x_dim)

get_name(p::StyblinskiTangProblem) = (p |> typeof |> string) * string(p.x_dim)


module StyblinskiTangProblemModule

import ..StyblinskiTangProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [0.0]
const std_obs = [5.0]

# --- API ---

simulator(p::StyblinskiTangProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::StyblinskiTangProblem)    = Domain(; bounds = (fill(-5.0, p.x_dim), fill(5.0, p.x_dim)))
likelihood(::StyblinskiTangProblem) = NormalLikelihood(; z_obs, std_obs)
prior_mean(::StyblinskiTangProblem) = z_obs
x_prior(p::StyblinskiTangProblem)   = Product(fill(Uniform(-5.0, 5.0), p.x_dim))
est_amplitude(::StyblinskiTangProblem)      = [50.0]
est_noise_std(::StyblinskiTangProblem)      = nothing
est_grad_noise_std(::StyblinskiTangProblem) = nothing
true_f(p::StyblinskiTangProblem)    = _true_f(p.x_dim)

# --- Simulator ---

_g(t) = 0.5 * (t^4 - 16.0*t^2 + 5.0*t)
_dg(t) = 2.0*t^3 - 16.0*t + 2.5

function _f(x)
    return [sum(_g, x)]
end

function _J(x)
    return reshape([_dg(xi) for xi in x], 1, length(x))
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module StyblinskiTangProblemModule
