"""
    StyblinskiTangProblem(; x_dim=2)

A d-dimensional problem with the Styblinski-Tang simulator.

Simulator: y = ½ Σᵢ₌₁ᵈ (xᵢ⁴ − 16xᵢ² + 5xᵢ)  (scalar output)

The function is separable: each dimension contributes independently. The 1D
component g(t) = ½(t⁴ − 16t² + 5t) has two local minima at t ≈ −2.903
(g ≈ −39.17, global) and t ≈ 2.747 (g ≈ −25.1). This separable structure
means the response surface has 2^d local minima in d dimensions.

For d=2, the four modes of f are at approximately (−2.90,−2.90) → y≈−78.3,
(−2.90,+2.75)/(+2.75,−2.90) → y≈−64.2, and (+2.75,+2.75) → y≈−50.2.
With z_obs = −60 and std_obs = 8, the likelihood balances the contributions
from all four modes, producing a clearly multimodal posterior. Good benchmark
for testing acquisition functions on multimodal posteriors and for illustrating
how a non-injective proxy (e.g. modeling |y| instead of y) loses information.
"""
@kwdef struct StyblinskiTangProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [8.0]
end

set_gradients(p::StyblinskiTangProblem, val::Bool) = StyblinskiTangProblem(val, p.x_dim, p.std_obs)

get_name(p::StyblinskiTangProblem) = (p |> typeof |> string) * string(p.x_dim)


module StyblinskiTangProblemModule

import ..StyblinskiTangProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [-3.248518, 3.084936]
const x_true_5 = [-3.248518, 3.084936, -3.248518, 3.084936, -3.248518]

# --- API ---

simulator(p::StyblinskiTangProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::StyblinskiTangProblem)    = Domain(; bounds = (fill(-5.0, p.x_dim), fill(5.0, p.x_dim)))
_z_obs(p::StyblinskiTangProblem) = true_f(p)(true_params(p))
likelihood(p::StyblinskiTangProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::StyblinskiTangProblem) = _z_obs(p)
x_prior(p::StyblinskiTangProblem)   = Product(fill(Uniform(-5.0, 5.0), p.x_dim))
est_amplitude(::StyblinskiTangProblem)      = [50.0]
est_noise_std(::StyblinskiTangProblem)      = nothing
est_grad_noise_std(::StyblinskiTangProblem) = nothing
true_f(p::StyblinskiTangProblem)    = _true_f(p.x_dim)
function true_params(p::StyblinskiTangProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

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
