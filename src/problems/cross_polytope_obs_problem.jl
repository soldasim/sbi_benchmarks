"""
    CrossPolytopeObsProblem(inner::AbstractProblem)

Wrapper that replaces a single observation at θ* with 2d observations placed at
the vertices of a cross-polytope of radius 2.5% of the domain width centred at
θ*, where d is the parameter dimension.

The 2d offsets are ±r·eᵢ for i = 1…d (one pair per coordinate axis). In 2D
this gives a square on its vertex (diamond, 4 obs); in 3D an octahedron (6 obs);
in 5D a 5-orthoplex (10 obs). The true parameters θ* are never directly observed.

Supports any dimensionality (unlike HexObsProblem which is 2D-only).

For plain `NormalLikelihood` problems the 2d-output likelihood is a
`NormalLikelihood` with independent entries and the same σ as the inner problem.
For `CustomLikelihood` proxy problems (log-proxy with offset `C = proxy_offset(inner)`)
the 2d-output likelihood is a `CustomLikelihood` whose `log_ψ` inverts the proxy
for each output independently and compares to pre-computed raw observations.

`get_name` appends "_cross" so data are stored separately from other variants.
"""

const _CROSS_RADIUS_FRACTION = 0.025   # 2.5% of minimum domain width

struct CrossPolytopeObsProblem{P <: AbstractProblem} <: AbstractProblem
    inner::P
    offsets::Vector{Vector{Float64}}   # length-2d list of ±eᵢ offset vectors
    z_obs_cross::Vector{Float64}        # 2d pre-computed observations (proxy or raw)
end

function CrossPolytopeObsProblem(inner::P) where {P <: AbstractProblem}
    lb, ub = domain(inner).bounds
    d = x_dim(inner)
    r = _CROSS_RADIUS_FRACTION * minimum(ub .- lb)
    offsets = Vector{Vector{Float64}}()
    for i in 1:d
        e = zeros(d); e[i] = r
        push!(offsets,  e)
        push!(offsets, -e)
    end
    x_c = true_params(inner)
    f   = true_f(inner)
    z_obs_cross = vcat([f(x_c .+ δ) for δ in offsets]...)
    return CrossPolytopeObsProblem{P}(inner, offsets, z_obs_cross)
end

# --- Name and gradient forwarding ---

get_name(p::CrossPolytopeObsProblem)                = get_name(p.inner) * "_cross"
set_gradients(p::CrossPolytopeObsProblem, val::Bool) = CrossPolytopeObsProblem(set_gradients(p.inner, val))

# --- Domain / prior (unchanged) ---

domain(p::CrossPolytopeObsProblem)  = domain(p.inner)
x_prior(p::CrossPolytopeObsProblem) = x_prior(p.inner)

# --- Simulator ---

function simulator(p::CrossPolytopeObsProblem)
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
# NormalLikelihood problems: 2d-output NormalLikelihood with the same σ.
# CustomLikelihood (proxy) problems: 2d-output CustomLikelihood that inverts the
# log-proxy independently for each output and compares to pre-computed raw obs.

function likelihood(p::CrossPolytopeObsProblem)
    n_obs = length(p.offsets)
    C = proxy_offset(p.inner)
    if isnothing(C)
        std_obs_vec = fill(p.inner.std_obs[1], n_obs)
        return NormalLikelihood(; z_obs = p.z_obs_cross, std_obs = std_obs_vec)
    else
        std_raw = p.inner.std_obs[1]
        raw_obs = [exp(z) - C for z in p.z_obs_cross]
        log_ψ   = (δ_vec, x) -> sum(logpdf(Normal(raw_obs[i], std_raw), exp(δ_vec[i]) - C) for i in 1:n_obs)
        return CustomLikelihood(; log_ψ, δ_dim = n_obs, mc_samples = 1000)
    end
end

# --- GP / model hints ---

prior_mean(p::CrossPolytopeObsProblem)        = p.z_obs_cross
est_amplitude(p::CrossPolytopeObsProblem)     = repeat(est_amplitude(p.inner), length(p.offsets))
est_noise_std(::CrossPolytopeObsProblem)      = nothing
est_grad_noise_std(::CrossPolytopeObsProblem) = nothing
y_max(::CrossPolytopeObsProblem)              = nothing

# --- Reference ---

function true_f(p::CrossPolytopeObsProblem)
    f_inner = true_f(p.inner)
    offsets = p.offsets
    return (x) -> vcat([f_inner(x .+ δ) for δ in offsets]...)
end

true_params(p::CrossPolytopeObsProblem) = true_params(p.inner)
