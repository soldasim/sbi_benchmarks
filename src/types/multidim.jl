
"""
    Multidimproblem(::AbstractProblem, scaleup::Int)

A wrapper that takes a problem and creates a new problem with `scaleup` times more input dimensions,
by replicating the original problem's input space. The outputs are the MEAN of the replicated simulators.

That is, if the original problem has input dimension `d`, the new problem will have input dimension `d * scaleup`,
but output dimension remains the same. The new simulator is defined as 
`f_multidim(x) = (f(x[1:d]) + f(x[d+1:2d]) + ... + f(x[(scaleup-1)*d+1:scaleup*d])) / scaleup`,
where `f` is the original simulator.
"""
struct MultidimProblem{
    T<:AbstractProblem,
} <: AbstractProblem
    problem::T
    scaleup::Int
    gradients::Bool

    function MultidimProblem(problem::T, scaleup::Int) where {T<:AbstractProblem}
        @assert scaleup > 0
        gradients = MultidimProblemModule._has_gradients(problem)
        new{T}(problem, scaleup, gradients)
    end
end

function set_gradients(p::MultidimProblem, val::Bool)
    base_problem = set_gradients(p.problem, val)
    return MultidimProblem(base_problem, p.scaleup)
end


module MultidimProblemModule

import ..AbstractProblem
import ..MultidimProblem

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

import ..y_dim

using BOSS
using BOSIP
using Distributions
using Random

include("../utils/combined_distribution.jl")


# --- API ---

# Helper function to check if base problem supports gradients
function _has_gradients(p::T) where {T<:AbstractProblem}
    hasfield(T, :gradients) && p.gradients
end

# Helper function to split input into chunks and apply function, averaging outputs
function _apply_multidim_sum(f::Function, x::AbstractVector, d::Int, scaleup::Int)
    y_sum = nothing
    for i in 1:scaleup
        x_i = @view x[((i-1)*d + 1):(i*d)]
        y_i = f(x_i)
        if y_sum === nothing
            y_sum = copy(y_i)
        else
            y_sum .+= y_i
        end
    end
    return y_sum / scaleup
end
function _apply_multidim_sum(f::Function, X::AbstractMatrix, d::Int, scaleup::Int)
    Y_sum = nothing
    for i in 1:scaleup
        X_i = @view X[((i-1)*d + 1):(i*d), :]
        Y_i = f(X_i)
        if Y_sum === nothing
            Y_sum = copy(Y_i)
        else
            Y_sum .+= Y_i
        end
    end
    return Y_sum / scaleup
end

# Helper function to split input and apply function that returns (y, J), averaging outputs and concatenating Jacobians
function _apply_multidim_sum_with_grads(f::Function, x::AbstractVector, d::Int, scaleup::Int)
    y_parts = []
    J_parts = []
    for i in 1:scaleup
        x_i = @view x[((i-1)*d + 1):(i*d)]
        y_i, J_i = f(x_i)
        push!(y_parts, y_i)
        push!(J_parts, J_i)
    end
    
    # Average outputs
    y_mean = sum(y_parts; init=zero(y_parts[1])) / scaleup
    
    # Concatenate Jacobians horizontally, scaled by 1/scaleup for averaging
    # Each base Jacobian is (m, d), combined is (m, d*scaleup)
    J_concat = hcat(J_parts...) / scaleup
    
    return y_mean, J_concat
end

function _apply_multidim_sum_with_grads(f::Function, X::AbstractMatrix, d::Int, scaleup::Int)
    # For matrix input, compute outputs for each column by averaging across chunks
    n_samples = size(X, 2)
    
    # First pass: get output dimension from one call
    sample_out, _ = f(X[1:d, 1])
    m = length(sample_out)
    
    Y_sum = zeros(m, n_samples)
    J_list = [zeros(m, d * scaleup) for _ in 1:n_samples]
    
    # Compute outputs and Jacobians
    for col in 1:n_samples
        J_blocks = []
        for i in 1:scaleup
            x_i = @view X[((i-1)*d + 1):(i*d), col]
            y_i, J_i = f(x_i)
            Y_sum[:, col] .+= y_i
            push!(J_blocks, J_i)
        end
        # Concatenate Jacobian blocks horizontally, scaled by 1/scaleup for averaging
        J_list[col] = hcat(J_blocks...) / scaleup
    end
    
    # Average the outputs
    Y_sum ./= scaleup
    
    return Y_sum, J_list
end

# Helper function to create a wrapper that applies multidim averaging
function _create_multidim_wrapper(f::Function, d::Int, scaleup::Int, include_grads::Bool)
    if include_grads
        # Wrapper returns (y, J) tuples - sum outputs and Jacobians
        function multidim_with_grads(x::AbstractVector)
            _apply_multidim_sum_with_grads(f, x, d, scaleup)
        end
        function multidim_with_grads(X::AbstractMatrix)
            _apply_multidim_sum_with_grads(f, X, d, scaleup)
        end
        return multidim_with_grads
    else
        # Wrapper returns only y - sum outputs
        function multidim_no_grads(x::AbstractVector)
            _apply_multidim_sum(f, x, d, scaleup)
        end
        function multidim_no_grads(X::AbstractMatrix)
            _apply_multidim_sum(f, X, d, scaleup)
        end
        return multidim_no_grads
    end
end

function simulator(p::MultidimProblem)
    base_f = simulator(p.problem)
    d = length(domain(p.problem).bounds[1])
    scaleup = p.scaleup
    include_grads = _has_gradients(p)
    
    return _create_multidim_wrapper(base_f, d, scaleup, include_grads)
end

function true_f(p::MultidimProblem)
    base_f = true_f(p.problem)
    isnothing(base_f) && return nothing
    
    d = length(domain(p.problem).bounds[1])
    scaleup = p.scaleup
    
    # true_f should always return results WITHOUT gradients
    return _create_multidim_wrapper(base_f, d, scaleup, false)
end

function domain(p::MultidimProblem)
    base_domain = domain(p.problem)
    bounds_lower = base_domain.bounds[1]
    bounds_upper = base_domain.bounds[2]
    
    new_bounds_lower = repeat(bounds_lower, p.scaleup)
    new_bounds_upper = repeat(bounds_upper, p.scaleup)
    
    return Domain(; bounds = (new_bounds_lower, new_bounds_upper))
end

function likelihood(p::MultidimProblem)
    # Since outputs are summed, y_dim doesn't increase, so use base likelihood directly
    likelihood(p.problem)
end

function prior_mean(p::MultidimProblem)
    # Since outputs are the mean, use base prior mean unchanged
    prior_mean(p.problem)
end

function x_prior(p::MultidimProblem)
    base_prior = x_prior(p.problem)
    # Create a combined distribution that properly samples vectors
    CombinedDistribution(base_prior, p.scaleup)
end

function est_amplitude(p::MultidimProblem)
    # Since outputs are the mean, use base amplitude unchanged
    est_amplitude(p.problem)
end

function est_noise_std(p::MultidimProblem)
    base_noise = est_noise_std(p.problem)
    isnothing(base_noise) && return nothing
    # When averaging independent random variables, variance reduces, so std scales by 1/sqrt(scaleup)
    base_noise / sqrt(p.scaleup)
end

function est_grad_noise_std(p::MultidimProblem)
    base_grad_noise = est_grad_noise_std(p.problem)
    isnothing(base_grad_noise) && return nothing
    # When averaging independent random variables, variance reduces, so std scales by 1/sqrt(scaleup)
    base_grad_noise / sqrt(p.scaleup)
end

function reference_samples(p::MultidimProblem)
    # Since outputs are summed, reference samples don't have a straightforward definition
    # We return nothing to indicate that true_f should be used instead
    nothing
end

function y_max(p::MultidimProblem)
    base_y_max = y_max(p.problem)
    isnothing(base_y_max) && return nothing
    # Since outputs are the mean, max value is unchanged
    base_y_max
end

end # module MultidimProblemModule
