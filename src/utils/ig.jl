
"""
Selects new data point by maximizing the information gain in the posterior.

The information gain is calculated as the mutual information between
the new data point (a vector-valued random variable from a multivariate distribution given by the GPs)
and the posterior approximation (a "random function" from a infinite-dimensional distribution).

Instead of calculating the information gain of the infinite-dimensional parameter
posterior function values, `InfoGainInt` computes the average information gain
integrated over the domain.

The resulting mutual information between the new data point
and the functional values of the approximate posterior
is estimated by calculating the maximum mean discrepancy (MMD)
between their joint and marginal distributions.
This is also known as  the Hilbert-Schmidt independence criterion (HSIC).

The random function (an infinite-dimensional random variable) representing
the posterior is reduces to samples, which are integrated over.
The samples are drawn from the `x_proposal`.

# Kwargs
- `x_samples::Int64`: The amount of samples used to approximate the integral
        over the parameter domain.
- `samples::Int64`: The amount of samples drawn from the joint and marginal distributions
        to estimate the HSIC value.
- `x_proposal::MultivariateDistribution`:
- `y_kernel::Kernel`: The kernel used for the samples of the new data point.
- `p_kernel::Kernel`: The kernel used for the posterior function value samples.
"""
@kwdef struct InfoGainInt <: BolfiAcquisition
    x_samples::Int64
    samples::Int64
    x_proposal::MultivariateDistribution
    y_kernel::BOSS.Kernel = BOSS.GaussianKernel()
    p_kernel::BOSS.Kernel = BOSS.GaussianKernel()
end

# info gain on the posterior approximation
function (acq::InfoGainInt)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions; return_xs=false)
    problem = bolfi.problem
    y_dim = BOSS.y_dim(problem)

    # Sample parameter values.
    xs = rand(acq.x_proposal, acq.x_samples)
    
    # w_i = 1 / pdf(x_proposal, x_i)
    log_ws = 0 .- logpdf.(Ref(acq.x_proposal), eachcol(xs))
    ws = exp.( log_ws .- log(sum(exp.(log_ws))) ) # normalize to sum up to 1

    # Sample noise variables (makes the resulting acquisition function deterministic)
    ϵs_y = sample_ϵs(y_dim, acq.samples) # vector-vector
    Es_s = [sample_ϵs(y_dim, acq.samples) for _ in 1:acq.x_samples] # vector-vector-vector

    # additional 

    # Compute HSIC
    a = construct_hsic_acquisition(acq, bolfi, xs, ws, ϵs_y, Es_s)
    if return_xs
        return a, xs
    else
        return a
    end
end

function sample_ϵs(y_dim, samples)
    d = MvNormal(zeros(y_dim), ones(y_dim))
    ϵs = [rand(d) for _ in 1:samples]
    return ϵs
end

# TODO: wrong -> claculate one HSIC for each x sample, importance weighting to counterbalance sampling from x_prior
function construct_hsic_acquisition(
    acq::InfoGainInt,
    bolfi::BolfiProblem,
    xs::AbstractMatrix{<:Real}, # x samples to integrate over the domain
    ws::AbstractVector{<:Real}, # weights for the x samples (importance sampling)
    ϵs_y::AbstractVector{<:AbstractVector{<:Real}}, # [GP] noise samples for sampling the GP posterior
    Es_s::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}}, # [DOMAIN × GP] noise samples for the posterior pdf values
)
    boss = bolfi.problem
    gp_post = BOSS.model_posterior(boss)

    @warn "Using experimental length-scales for the HSIC kernels."

    function acq_(x_)
        # sample `N` y_ samples at the new x_
        μy, σy = gp_post(x_)
        ys_ = calc_y.(Ref(μy), Ref(σy), ϵs_y)

        # augment problems
        problem_copies = [deepcopy(boss) for _ in 1:acq.samples]
        for (p, y_) in zip(problem_copies, ys_)
            augment_dataset!(p, x_, y_)
        end
        aug_posts_samples = model_posterior.(problem_copies)

        # sample `K x N` y_eval (and s_eval) samples (N for each x_eval sample)
        Y_evals = get_ys_eval.(Ref(aug_posts_samples), eachcol(xs), Es_s) # vector-vector-vector
        # S_evals = get_ss_eval.(Y_evals, eachcol(xs), Ref(bolfi.likelihood), Ref(bolfi.x_prior))
        log_S_evals = get_log_ss_eval.(Y_evals, eachcol(xs), Ref(bolfi.likelihood), Ref(bolfi.x_prior)) # vector-vector

        # S_evals are UNNORMALIZED! (This is intentional, as there is no straight-forward way to normalize them.)
        
        # calculate `K` HSICs between `y_` and `s_1,...,s_K`
        
        # y lengthscale
        y_lengthscale = σy
        y_kernel = BOSS.with_lengthscale(acq.y_kernel, y_lengthscale)

        # s lengthscales
        # (normalization instead of calculating different lengthscales)
        p_kernel = acq.p_kernel
        normalize_log_S_evals!(log_S_evals)
        S_evals = exponentiate_S_evals!(log_S_evals) # in-place exponentiation
        
        vals = hsic.(Ref(y_kernel), Ref(p_kernel), Ref(ys_), S_evals, Ref(acq.samples))

        return sum(ws .* vals)
    end
end

function calc_y(μ, σ, ϵ)
    return μ .+ (σ .* ϵ)
end

function get_ys_eval(aug_posts, x_eval, ϵs)
    pred_distrs = [aug_posts[i](x_eval) for i in eachindex(aug_posts)]
    ys_eval = [calc_y(pred_distrs[i]..., ϵs[i]) for i in eachindex(pred_distrs)]
    return ys_eval
end
# function get_ss_eval(ys_eval, x_eval, likelihood, x_prior)
#     log_ls = BOLFI.loglike.(Ref(likelihood), ys_eval)
#     log_px = logpdf(x_prior, x_eval)
#     ss = exp.(log_px .+ log_ls) # TODO unnormalized
#     return ss
# end
function get_log_ss_eval(ys_eval, x_eval, likelihood, x_prior)
    log_ls = BOLFI.loglike.(Ref(likelihood), ys_eval)
    log_px = logpdf(x_prior, x_eval)
    ss = log_ls .+ log_px # TODO unnormalized
    return ss
end

# function mmd(kernel, X::AbstractVector, Y::AbstractVector)
#     val_X = mean(BOSS.kernelmatrix(kernel, X))
#     val_Y = mean(BOSS.kernelmatrix(kernel, Y))
#     val_XY = mean(BOSS.kernelmatrix(kernel, X, Y))
#     return val_X + val_Y - 2*val_XY
# end
function hsic(kernel_X, kernel_Y, X::AbstractVector, Y::AbstractVector, n::Int)
    Kx = BOSS.kernelmatrix(kernel_X, X)
    Ky = BOSS.kernelmatrix(kernel_Y, Y)
    C = Diagonal(ones(n)) - (1/n) * ones(n,n)
    return (1 / (n^2)) * tr( (C * Kx) * (C * Ky) )
end

function normalize_log_S_evals!(log_S_evals)
    log_M = maximum(maximum.(log_S_evals))
    for i in eachindex(log_S_evals)
        for j in eachindex(log_S_evals[i])
            log_S_evals[i][j] -= log_M
        end
    end
    return log_S_evals
end

function exponentiate_S_evals!(log_S_evals)
    for ix in eachindex(log_S_evals)
        log_S_evals[ix] = exp.(log_S_evals[ix])
    end
    return log_S_evals # S_evals
end
