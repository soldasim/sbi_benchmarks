"""
    SharpProblem(base::AbstractProblem)

Wrapper that scales the observation std_obs of any opt-function problem by `SHARP_FACTOR`,
giving a sharper (more concentrated) likelihood while keeping everything else identical.

`get_name` appends "_sharp" so data are stored separately from the default problems.
"""

const SHARP_FACTOR = 0.04

# Scale std_obs of any @kwdef problem struct by SHARP_FACTOR at construction time.
function _apply_sharp(p)
    T = typeof(p)
    fields = fieldnames(T)
    kwargs = NamedTuple{fields}(ntuple(length(fields)) do i
        fields[i] === :std_obs ? getfield(p, :std_obs) .* SHARP_FACTOR : getfield(p, fields[i])
    end)
    return T(; kwargs...)
end

struct SharpProblem{T <: AbstractProblem} <: AbstractProblem
    base::T
    SharpProblem(p::T) where {T <: AbstractProblem} = new{T}(_apply_sharp(p))
end

get_name(p::SharpProblem)                      = get_name(p.base) * "_sharp"
set_gradients(p::SharpProblem, val::Bool)      = SharpProblem(set_gradients(_unsharp(p.base), val))
simulator(p::SharpProblem)                     = simulator(p.base)
domain(p::SharpProblem)                        = domain(p.base)
likelihood(p::SharpProblem)                    = likelihood(p.base)
prior_mean(p::SharpProblem)                    = prior_mean(p.base)
x_prior(p::SharpProblem)                       = x_prior(p.base)
est_amplitude(p::SharpProblem)                 = est_amplitude(p.base)
est_noise_std(p::SharpProblem)                 = est_noise_std(p.base)
est_grad_noise_std(p::SharpProblem)            = est_grad_noise_std(p.base)
true_f(p::SharpProblem)                        = true_f(p.base)
reference_samples(p::SharpProblem)             = reference_samples(p.base)
y_max(p::SharpProblem)                         = y_max(p.base)

# Reconstruct the unscaled base problem (needed in set_gradients to avoid double-scaling).
function _unsharp(p)
    T = typeof(p)
    fields = fieldnames(T)
    kwargs = NamedTuple{fields}(ntuple(length(fields)) do i
        fields[i] === :std_obs ? getfield(p, :std_obs) ./ SHARP_FACTOR : getfield(p, fields[i])
    end)
    return T(; kwargs...)
end
