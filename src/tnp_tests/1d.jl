include("../main.jl")

using Random
using LinearAlgebra
using Distributions
using ProgressMeter
using CairoMakie

# Simple 1D GP-sampled objective problem
struct GP1DProblem <: AbstractProblem
    f::Function
    bounds::Tuple{Float64, Float64}
end

function sample_gp_problem()
    bounds = (0.0, 1.0)

    # Sample a 1D objective from a GP prior
    f_sample = sample_gp_function(; bounds)
    problem = GP1DProblem(f_sample, bounds)

    return problem
end

# --- Problem API ---
simulator(p::GP1DProblem) = x -> [p.f(x)]
domain(p::GP1DProblem) = Domain(; bounds = ([p.bounds[1]], [p.bounds[2]]))
likelihood(::GP1DProblem) = NormalLikelihood(; z_obs = [0.0], std_obs = [1.0])
prior_mean(::GP1DProblem) = [0.0]
x_prior(p::GP1DProblem) = product_distribution([Uniform(p.bounds...)])
est_amplitude(::GP1DProblem) = [1.0]
est_noise_std(::GP1DProblem) = nothing
true_f(p::GP1DProblem) = x -> [p.f(x)]
get_lengthscale_priors(::GP1DProblem) = fill(product_distribution([LogNormal(log(0.2), 0.5)]), 1)
get_amplitude_priors(::GP1DProblem) = fill(LogNormal(log(1.0), 0.5), 1)
get_noise_std_priors(::GP1DProblem) = fill(LogNormal(log(0.05), 0.5), 1)


# Sample a smooth 1D function from a zero-mean Matérn 5/2 GP on [bounds...]
function sample_gp_function(; bounds::Tuple{Float64, Float64} = (0.0, 1.0), n_points::Int = 400,
    amplitude::Float64 = 1.0, ℓ::Float64 = 0.2, jitter::Float64 = 1e-6, rng = Random.default_rng())

    xs = collect(range(bounds[1], bounds[2], length = n_points))
    matern52(r) = (1 + sqrt(5) * r / ℓ + 5 * r^2 / (3 * ℓ^2)) * exp(-sqrt(5) * r / ℓ)

    K = Matrix{Float64}(undef, n_points, n_points)
    for i in 1:n_points
        K[i, i] = amplitude^2 + jitter
        for j in (i + 1):n_points
            r = abs(xs[i] - xs[j])
            val = amplitude^2 * matern52(r)
            K[i, j] = val
            K[j, i] = val
        end
    end

    y = rand(rng, MvNormal(zeros(n_points), Symmetric(K)))

    # lightweight linear interpolant
    function f_interp(x::AbstractVector{<:Real})
        xx = clamp(x[1], bounds[1], bounds[2])
        idx = searchsortedfirst(xs, xx)
        if idx == 1
            return y[1]
        elseif idx > length(xs)
            return y[end]
        else
            x0, x1 = xs[idx - 1], xs[idx]
            y0, y1 = y[idx - 1], y[idx]
            t = (xx - x0) / (x1 - x0)
            return (1 - t) * y0 + t * y1
        end
    end

    return f_interp
end


# estimator: "mean" or "approx"
function tnp_test_1d()
    bounds = (0.0, 1.0)

    # Sample a 1D objective from a GP prior
    f_sample = sample_gp_function(; bounds)
    problem = GP1DProblem(f_sample, bounds)
    sim = simulator(problem)

    # Training data (sparse grid)
    # TODO
    # n_train = 15
    # x_train = collect(range(bounds[1], bounds[2], length = n_train))
    x_train = [0.4, 0.5, 0.6]

    X_train = reshape(x_train, 1, :)
    Y_train = [sim([x])[1] for x in x_train]'

    data = BOSS.ExperimentData(X_train, Y_train)

    # Initialize both models
    tnp_model = TNP()
    gp_model = GaussianProcess(;
        mean = prior_mean(problem),
        kernel = BOSS.Matern52Kernel(),
        lengthscale_priors = get_lengthscale_priors(problem),
        amplitude_priors = get_amplitude_priors(problem),
        noise_std_priors = get_noise_std_priors(problem),
    )

    # Build BosipProblems for both models
    acquisition = LogMaxVar()
    
    bosip_gp = construct_bosip_problem(;
        problem,
        data,
        acquisition,
        model = gp_model,
    )
    
    bosip_tnp = construct_bosip_problem(;
        problem,
        data,
        acquisition,
        model = tnp_model,
    )

    # Fit model hyperparameters
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 12,
        parallel = false,
        rhoend = 1e-4,
    )
    
    println("Fitting GP model...")
    BOSIP.estimate_parameters!(bosip_gp, model_fitter)
    
    println("Fitting TNP model...")
    BOSIP.estimate_parameters!(bosip_tnp, model_fitter)

    model_posterior_gp = BOSS.model_posterior(bosip_gp.problem)
    model_posterior_tnp = BOSS.model_posterior(bosip_tnp.problem)

    # Get posterior mean and variance functions
    logpost_mean_gp = BOSIP.log_posterior_mean(bosip_gp)
    logpost_var_gp = BOSIP.log_posterior_variance(bosip_gp)
    logpost_mean_tnp = BOSIP.log_posterior_mean(bosip_tnp)
    logpost_var_tnp = BOSIP.log_posterior_variance(bosip_tnp)

    # Evaluation grid
    n_test = 400
    x_test = collect(range(bounds[1], bounds[2], length = n_test))

    y_true = similar(x_test)
    y_mean_gp = similar(x_test)
    y_std_gp = similar(x_test)
    y_mean_tnp = similar(x_test)
    y_std_tnp = similar(x_test)
    post_true = similar(x_test)
    post_mean_gp = similar(x_test)
    post_std_gp = similar(x_test)
    post_mean_tnp = similar(x_test)
    post_std_tnp = similar(x_test)

    logpost_true = true_logpost(problem)

    @showprogress desc = "Evaluating" for (i, x) in enumerate(x_test)
        y_true[i] = sim([x])[1]
        
        m_gp, v_gp = mean_and_var(model_posterior_gp, [x])
        y_mean_gp[i] = m_gp[1]
        y_std_gp[i] = sqrt(v_gp[1])
        
        m_tnp, v_tnp = mean_and_var(model_posterior_tnp, [x])
        y_mean_tnp[i] = m_tnp[1]
        y_std_tnp[i] = sqrt(v_tnp[1])

        post_true[i] = exp(logpost_true([x]))
        post_mean_gp[i] = exp(logpost_mean_gp([x]))
        post_std_gp[i] = sqrt(exp(logpost_var_gp([x])))
        post_mean_tnp[i] = exp(logpost_mean_tnp([x]))
        post_std_tnp[i] = sqrt(exp(logpost_var_tnp([x])))
    end

    # Plotting - 2x3 grid
    fig = Figure(size = (1200, 800))
    std_scale = 2.0

    z_obs = likelihood(problem).z_obs[1]

    # Column 1: True
    ax1 = Axis(fig[1, 1]; title = "True function (GP sample)", xlabel = "x", ylabel = "f(x)")
    lines!(ax1, x_test, y_true; color = :black)
    hlines!(ax1, [z_obs]; color = :green, linestyle = :dash, label = "z_obs")
    scatter!(ax1, x_train, vec(Y_train); color = :red, label = "train")

    ax2 = Axis(fig[2, 1]; title = "True posterior", xlabel = "x", ylabel = "p(x)")
    lines!(ax2, x_test, post_true; color = :black)

    # Column 2: GP
    ax3 = Axis(fig[1, 2]; title = "GP mean ± $std_scale std", xlabel = "x", ylabel = "f(x)")
    lines!(ax3, x_test, y_mean_gp; color = :blue)
    band!(ax3, x_test, y_mean_gp .- std_scale .* y_std_gp, y_mean_gp .+ std_scale .* y_std_gp; color = (:blue, 0.15))
    hlines!(ax3, [z_obs]; color = :green, linestyle = :dash, label = "z_obs")
    scatter!(ax3, x_train, vec(Y_train); color = :red)

    ax4 = Axis(fig[2, 2]; title = "GP posterior mean ± $std_scale std", xlabel = "x", ylabel = "p(x)")
    lines!(ax4, x_test, post_mean_gp; color = :blue)
    band!(ax4, x_test, post_mean_gp .- std_scale .* post_std_gp, post_mean_gp .+ std_scale .* post_std_gp; color = (:blue, 0.15))

    # Column 3: TNP
    ax5 = Axis(fig[1, 3]; title = "TNP mean ± $std_scale std", xlabel = "x", ylabel = "f(x)")
    lines!(ax5, x_test, y_mean_tnp; color = :orange)
    band!(ax5, x_test, y_mean_tnp .- std_scale .* y_std_tnp, y_mean_tnp .+ std_scale .* y_std_tnp; color = (:orange, 0.15))
    hlines!(ax5, [z_obs]; color = :green, linestyle = :dash, label = "z_obs")
    scatter!(ax5, x_train, vec(Y_train); color = :red)

    ax6 = Axis(fig[2, 3]; title = "TNP posterior mean ± $std_scale std", xlabel = "x", ylabel = "p(x)")
    lines!(ax6, x_test, post_mean_tnp; color = :orange)
    band!(ax6, x_test, post_mean_tnp .- std_scale .* post_std_tnp, post_mean_tnp .+ std_scale .* post_std_tnp; color = (:orange, 0.15))

    save("plots/test_1d.png", fig)
    fig
end
