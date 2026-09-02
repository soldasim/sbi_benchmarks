"""
    HexObsProblem(inner::AbstractProblem)

Wrapper that replaces a single observation at θ* with 6 observations on a hexagonal
ring of radius 2.5% of the domain width centred at θ*.

The simulator maps θ → [f(θ+δ₁), …, f(θ+δ₆)] where δᵢ are the six hexagonal
offsets. The true parameters (the centre θ*) are never directly observed; only the
six surrounding outputs are. This over-specifies the 2D parameter vector with 6
observations, producing a sharper posterior than the single-observation version.

Only 2D inner problems are supported (throws otherwise).

For plain `NormalLikelihood` problems the 6-output likelihood is also a
`NormalLikelihood` with independent entries and the same σ as the inner problem.
For `CustomLikelihood` proxy problems (log-proxy with offset `C = proxy_offset(inner)`)
the 6-output likelihood is a `CustomLikelihood` whose `log_ψ` inverts the proxy for
each output independently and compares to the pre-computed raw observations.

`get_name` appends "_hex" so data are stored separately.
"""

const _HEX_RADIUS_FRACTION = 0.025   # 2.5% of minimum domain width

struct HexObsProblem{P <: AbstractProblem} <: AbstractProblem
    inner::P
    offsets::Vector{Vector{Float64}}   # length-6 list of 2D offset vectors
    z_obs_hex::Vector{Float64}          # 6 pre-computed observations (proxy or raw)
end

function HexObsProblem(inner::P) where {P <: AbstractProblem}
    x_dim(inner) == 2 || error(
        "HexObsProblem only supports 2D inner problems; got $(x_dim(inner))D $(typeof(inner))"
    )
    lb, ub = domain(inner).bounds
    r = _HEX_RADIUS_FRACTION * minimum(ub .- lb)
    offsets = [r .* [cos(k * π / 3), sin(k * π / 3)] for k in 0:5]
    x_c = true_params(inner)
    f   = true_f(inner)
    z_obs_hex = vcat([f(x_c .+ δ) for δ in offsets]...)
    return HexObsProblem{P}(inner, offsets, z_obs_hex)
end

# --- Name and gradient forwarding ---

get_name(p::HexObsProblem)             = get_name(p.inner) * "_hex"
set_gradients(p::HexObsProblem, val::Bool) = HexObsProblem(set_gradients(p.inner, val))

# --- Domain / prior (unchanged) ---

domain(p::HexObsProblem)  = domain(p.inner)
x_prior(p::HexObsProblem) = x_prior(p.inner)

# --- Simulator ---

function simulator(p::HexObsProblem)
    inner_sim = simulator(p.inner)
    offsets   = p.offsets
    if p.inner.gradients
        return (x) -> begin
            results = [inner_sim(x .+ δ) for δ in offsets]
            f_vals  = vcat([r[1] for r in results]...)
            J       = vcat([r[2] for r in results]...)
            return (f_vals, J)
        end
    else
        return (x) -> vcat([inner_sim(x .+ δ) for δ in offsets]...)
    end
end

# --- Likelihood ---
#
# NormalLikelihood problems: 6-output NormalLikelihood with the same σ.
# CustomLikelihood (proxy) problems: 6-output CustomLikelihood that inverts the
# log-proxy independently for each output and compares to pre-computed raw obs.

function likelihood(p::HexObsProblem)
    C = proxy_offset(p.inner)
    if isnothing(C)
        std_obs_vec = fill(p.inner.std_obs[1], 6)
        return NormalLikelihood(; z_obs = p.z_obs_hex, std_obs = std_obs_vec)
    else
        std_raw  = p.inner.std_obs[1]
        raw_obs  = [exp(z) - C for z in p.z_obs_hex]
        log_ψ    = (δ_vec, x) -> sum(logpdf(Normal(raw_obs[i], std_raw), exp(δ_vec[i]) - C) for i in 1:6)
        return CustomLikelihood(; log_ψ, δ_dim = 6, mc_samples = 1000)
    end
end

# --- GP / model hints ---

prior_mean(p::HexObsProblem)          = p.z_obs_hex
est_amplitude(p::HexObsProblem)       = repeat(est_amplitude(p.inner), 6)
est_noise_std(::HexObsProblem)        = nothing
est_grad_noise_std(::HexObsProblem)   = nothing
y_max(::HexObsProblem)                = nothing

# --- Reference ---

function true_f(p::HexObsProblem)
    f_inner = true_f(p.inner)
    offsets = p.offsets
    return (x) -> vcat([f_inner(x .+ δ) for δ in offsets]...)
end

true_params(p::HexObsProblem) = true_params(p.inner)
