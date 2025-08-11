
"""
    LogSIRProblem()

"Determinized" SIR problem from the SBIBM benchmark.

Only the log-likelihood is returned by the simulator.
See also the `SIRProblem` for a version of the problem,
where the simulator returns the probability `p` directly.
"""
struct LogSIRProblem <: AbstractProblem end

module SIRModule

import ..LogSIRProblem

import ..simulator
import ..domain
import ..y_max
import ..likelihood
import ..prior_mean
import ..x_prior
import ..est_amplitude
import ..est_noise_std
import ..true_f
import ..reference_samples

using BOSS
using BOLFI
using Distributions
using PyCall
using PythonCall


### API

# TODO noise
# (not using noise in order to compare with loglike modeling more fairly)

function _sir(x)
    x_ = collect(x)
    y_ = py"sir_simulator"(x_)
    y = pyconvert(Matrix{Float64}, y_)[1,:]
    return y
end

# TODO loglike
# function simulator(::LogSIRProblem)
#     return _sir
# end
function simulator(::LogSIRProblem)
    function _sir_loglike(x)
        y = _sir(x)

        y .= clamp.(y, 0., 1.)
        # ll = sum(logpdf.(Binomial.(trials, z), z_obs))
        ll = mapreduce((t, p, y) -> logpdf(Binomial(t, p), y), +, trials, y, z_obs)
        return [ll]
    end
end

function domain(::LogSIRProblem)
    return Domain(; bounds)
end

# TODO loglike
# function likelihood(::LogSIRProblem)
#     return BinomialLikelihood(;
#         z_obs = Int64.(z_obs),
#         trials,
#         int_grid_size = 200,
#     )
# end
function likelihood(::LogSIRProblem)
    return ExpLikelihood()
end

# TODO loglike
# function prior_mean(p::LogSIRProblem)
#     ps = z_obs ./ likelihood(p).trials
#     return ps
# end
function prior_mean(p::LogSIRProblem)
    return [0.]
end

function x_prior(::LogSIRProblem)
    return product_distribution([
        truncated(LogNormal(-0.9163, 0.5); lower=bounds[1][1], upper=bounds[2][1]),
        truncated(LogNormal(-2.0794, 0.2); lower=bounds[1][2], upper=bounds[2][2]),
    ])
end

# TODO loglike
# est_amplitude(::LogSIRProblem) = fill(1., y_dim)
est_amplitude(::LogSIRProblem) = fill(1000., y_dim) # TODO ???

# TODO noise
est_noise_std(::LogSIRProblem) = nothing

function reference_samples(::LogSIRProblem)
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

const trials = fill(1000, 10)

end # end module SIRModule
