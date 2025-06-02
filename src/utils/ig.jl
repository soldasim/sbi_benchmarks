
"""
Selects new data point by maximizing the information gain in the posterior.

The information gain is calculated as the mutual information between
the new data point (a vector-valued random variable from a multivariate distribution given by the GPs)
and the posterior approximation (a "random function" from a infinite-dimensional distribution).

Instead of calculating the information gain of the infinite-dimensional parameter
posterior function values, `InfoGainInt` computes the average info gain
integrated over the domain.

The resulting mutual information between the new data point
and the functional values of the approximate posterior evaluated at `x_prior`
is estimated by calculating the maximum mean discrepancy (MMD)
between their joint and marginal distributions.
This is also known as  the Hilbert-Schmidt independence criterion (HSIC).

# Kwargs
- `x_samples::Int64`: The amount of samples used to approximate the integral
        over the parameter domain.
- `samples::Int64`: The amount of samples drawn from the joint distribution
        to estimate the HSIC value, and the amount of samples drawn from the prior
        to average the information gain over.
- `x_prior::MultivariateDistribution`:
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

    # Sample noise variables (makes the resulting acquisition function deterministic)
    ϵs_y = sample_ϵs(y_dim, acq.samples)
    Es_s = sample_Es(y_dim, acq.x_samples, acq.samples)

    # Compute HSIC
    a = construct_hsic_acquisition(acq, bolfi, xs, ϵs_y, Es_s)
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

function sample_Es(y_dim, x_samples, samples)
    d = MvNormal(zeros(y_dim), ones(y_dim))
    Es = [[rand(d) for _ in 1:samples] for _ in 1:x_samples]
    return Es
end

# TODO: wrong -> claculate one HSIC for each x sample, importance weighting to counterbalance sampling from x_prior
function construct_hsic_acquisition(
    acq::InfoGainInt,
    bolfi::BolfiProblem,
    xs::AbstractMatrix{<:Real},
    ϵs_y::AbstractVector{<:AbstractVector{<:Real}},
    Es_s::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}},
)
    boss = bolfi.problem
    gp_post = BOSS.model_posterior(boss)

    # # TODO remove
    # xs = [0.;0.;;]

    @warn "Using experimental length-scales for the HSIC kernels."
    # θ, λ, α, noise = bolfi.problem.data.params
    likelihood = bolfi.likelihood
    x_prior = bolfi.x_prior
    
    # ys_eval_ = map(x -> gp_post(x)[1], eachcol(xs))
    ys_eval_ = [calc_y(likelihood, gp_post(x)..., 0.) for x in eachcol(xs)]
    log_ps_eval_ = BOLFI.loglike.(Ref(likelihood), ys_eval_) .+ logpdf.(Ref(x_prior), eachcol(xs))
    log_M = maximum(log_ps_eval_)
    std_ps_scaled = std(exp.( log_ps_eval_ .- log_M ))

    function acq_(x_)
        # sample `N` y_ samples at the new x_
        μy, σy = gp_post(x_)
        ys_ = calc_y.(Ref(bolfi.likelihood), Ref(μy), Ref(σy), ϵs_y)

        # augment problems
        problem_copies = [deepcopy(boss) for _ in 1:acq.samples]
        for (p, y_) in zip(problem_copies, ys_)
            augment_dataset!(p, x_, y_)
        end
        aug_posts_samples = model_posterior.(problem_copies)

        # sample `K x N` y_eval (and s_eval) samples (N for each x_eval sample)
        Y_evals = get_ys_eval.(Ref(bolfi.likelihood), Ref(aug_posts_samples), eachcol(xs), Es_s)
        # S_evals = get_ss_eval.(Y_evals, eachcol(xs), Ref(bolfi.likelihood), Ref(bolfi.x_prior))
        log_S_evals = get_log_ss_eval.(Y_evals, eachcol(xs), Ref(bolfi.likelihood), Ref(bolfi.x_prior))

        # S_evals are UNNORMALIZED! (This is intentional, as there is no straight-forward way to normalize them.)

        # w_i = 1 / pdf(x_proposal, x_i)
        ws = exp.(0 .- logpdf.(Ref(acq.x_proposal), eachcol(xs)))
        
        # calculate `K` HSICs between `y_` and `s_1,...,s_K`
        
        # TODO experimental
        y_lengthscale = std(ys_)
        y_kernel = BOSS.with_lengthscale(acq.y_kernel, y_lengthscale)

        # TODO experimental
        # p_kernel = BOSS.with_lengthscale(acq.p_kernel, p_lengthscale)
        # scale probabilites instead of using a lengthscale (numerically more stable)
        p_kernel = acq.p_kernel
        S_evals = normalize_S_evals!(log_S_evals, log_M, std_ps_scaled)
        
        vals = hsic.(Ref(y_kernel), Ref(p_kernel), Ref(ys_), S_evals, Ref(acq.samples))
        # vals = hsic.(Ref(acq.y_kernel), Ref(acq.p_kernel), Ref(ys_), S_evals, Ref(acq.samples))

        return sum(ws .* vals)
    end
end

# TODO
function calc_y(::NormalLikelihood, μ, σ, ϵ)
    return μ .+ (σ .* ϵ)
end
function calc_y(::LogNormalLikelihood, μ, σ, ϵ)
    return μ .+ (σ .* ϵ)
end
function calc_y(::BinomialLikelihood, μ, σ, ϵ)
    y = ϵ # ~ Normal(0, 1)
    y = cdf.(Ref(Normal()), y) # ~ Uniform(0, 1)
    y = map((μ, σ, y) -> quantile(truncated(Normal(μ, σ); lower=0., upper=1.), y), μ, σ, y) # ~ truncated(Normal(μ, σ))
    return y
end

function get_ys_eval(likelihood, aug_posts, x_eval, ϵs)
    pred_distrs = [aug_posts[i](x_eval) for i in eachindex(aug_posts)]
    ys_eval = [calc_y(likelihood, pred_distrs[i]..., ϵs[i]) for i in eachindex(pred_distrs)]
    return ys_eval
end
function get_ss_eval(ys_eval, x_eval, likelihood, x_prior)
    log_ls = BOLFI.loglike.(Ref(likelihood), ys_eval)
    log_px = logpdf(x_prior, x_eval)
    ss = exp.(log_px .+ log_ls) # TODO unnormalized
    return ss
end
function get_log_ss_eval(ys_eval, x_eval, likelihood, x_prior)
    log_ls = BOLFI.loglike.(Ref(likelihood), ys_eval)
    log_px = logpdf(x_prior, x_eval)
    ss = log_px .+ log_ls # TODO unnormalized
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

function normalize_S_evals!(log_S_evals, log_M, std_ps_scaled)
    for i in eachindex(log_S_evals)
        for j in eachindex(log_S_evals[i])
            log_S_evals[i][j] = exp(log_S_evals[i][j] - log_M) / std_ps_scaled
        end
    end
    return log_S_evals # already exponentiated
end
