
"""
    SIRProblem()

"Determinized" SIR problem from the SBIBM benchmark.

The simulator noise is removed and the probability `p` (parameter of the binomila likelihood)
is returned from the simulator directly.
"""
struct SIRProblem <: AbstractProblem end

module SIRModule

import ..SIRProblem

import ..simulator
import ..domain
import ..y_max
import ..likelihood
import ..prior_mean
import ..x_prior
import ..y_extrema
import ..noise_std_priors
import ..true_f
import ..reference_samples

using BOSS
using BOLFI
using Distributions
using PyCall
using PythonCall


### API

function simulator(::SIRProblem)
    function sir(x)
        x_ = collect(x)
        y_ = py"sir_simulator"(x_)
        y = pyconvert(Matrix{Float64}, y_)[1,:]
        return y
    end
end

function domain(::SIRProblem)
    return Domain(; bounds)
end

function likelihood(::SIRProblem)
    return BinomialLikelihood(;
        z_obs = Int64.(z_obs),
        trials = fill(1000, 10),
        int_grid_size = 200,
    )
end

function prior_mean(p::SIRProblem)
    ps = z_obs ./ likelihood(p).trials
    return ps
end

function x_prior(::SIRProblem)
    return product_distribution([
        truncated(LogNormal(-0.9163, 0.5); lower=bounds[1][1], upper=bounds[2][1]),
        truncated(LogNormal(-2.0794, 0.2); lower=bounds[1][2], upper=bounds[2][2]),
    ])
end

function y_extrema(::SIRProblem)
    return fill(0., y_dim), fill(1., y_dim)
end

function noise_std_priors(::SIRProblem)
    return fill(Dirac(0.), y_dim)
end

function reference_samples(::SIRProblem)
    return ref_samples
end


### UTILS

py"""
import sbibm
import torch
"""

@pyinclude("src/problems/sbibm/bounds.py")
@pyinclude("src/problems/sbibm/sir_det/task.py")

py"""
sir_task = SIR()
sir_name = sir_task.name

sir_prior = sir_task.get_prior()
sir_simulator = sir_task.get_simulator()
sir_observation = sir_task.get_observation(num_observation=1)  # 10 per task
sir_true_params = sir_task.get_true_parameters(num_observation=1) # unused

sir_reference_samples = sir_task.get_reference_posterior_samples(num_observation=1)
"""

function _get_bounds()
    lb, ub = py"get_bounds(sir_task)"
    bounds = (pyconvert(Vector{Float64}, lb), pyconvert(Vector{Float64}, ub))
    return bounds
end

const z_obs = pyconvert(Matrix{Float64}, py"sir_observation")[1,:]
const y_dim = length(z_obs)
const bounds = _get_bounds()
const ref_samples = pyconvert(Matrix{Float64}, py"sir_reference_samples")' |> collect

end # end module SIRModule
