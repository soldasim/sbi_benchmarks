include("../main.jl")

using Random
using LinearAlgebra
using Distributions
using ProgressMeter
using CairoMakie

# 2D GP-sampled objective problem with 2 output dimensions
struct GP2DProblemMultiOutput <: AbstractProblem
    f1::Function  # First output function
    f2::Function  # Second output function
    bounds::Tuple{NTuple{2, Float64}, NTuple{2, Float64}}
end

function sample_gp_problem_2d_multioutput()
    bounds = ((0.0, 0.0), (1.0, 1.0))

    # Sample two independent 2D objectives from a GP prior
    f1_sample = sample_gp_function_2d(; bounds)
    f2_sample = sample_gp_function_2d(; bounds)
    problem = GP2DProblemMultiOutput(f1_sample, f2_sample, bounds)

    return problem
end

# --- Problem API ---
simulator(p::GP2DProblemMultiOutput) = x -> [p.f1(x); p.f2(x)]
domain(p::GP2DProblemMultiOutput) = Domain(; bounds = ([p.bounds[1]...], [p.bounds[2]...]))
likelihood(::GP2DProblemMultiOutput) = NormalLikelihood(; z_obs = [0.0, 0.0], std_obs = [1.0, 1.0])
prior_mean(::GP2DProblemMultiOutput) = [0.0, 0.0]
x_prior(p::GP2DProblemMultiOutput) = product_distribution([Uniform(p.bounds[1][i], p.bounds[2][i]) for i in 1:2])
est_amplitude(::GP2DProblemMultiOutput) = [1.0, 1.0]
est_noise_std(::GP2DProblemMultiOutput) = nothing
true_f(p::GP2DProblemMultiOutput) = x -> [p.f1(x); p.f2(x)]
get_lengthscale_priors(::GP2DProblemMultiOutput) = fill(product_distribution([LogNormal(log(0.2), 0.5) for _ in 1:2]), 2)
get_amplitude_priors(::GP2DProblemMultiOutput) = fill(LogNormal(log(1.0), 0.5), 2)
get_noise_std_priors(::GP2DProblemMultiOutput) = fill(LogNormal(log(0.05), 0.5), 2)


# Sample a smooth 2D function from a zero-mean Matérn 5/2 GP on [bounds...]
function sample_gp_function_2d(; bounds::Tuple{NTuple{2, Float64}, NTuple{2, Float64}} = ((0.0, 0.0), (1.0, 1.0)), 
    n_points::Int = 40, amplitude::Float64 = 1.0, ℓ::Float64 = 0.2, jitter::Float64 = 1e-6, rng = Random.default_rng())

    x1s = collect(range(bounds[1][1], bounds[2][1], length = n_points))
    x2s = collect(range(bounds[1][2], bounds[2][2], length = n_points))
    
    n_total = n_points * n_points
    xs = Matrix{Float64}(undef, 2, n_total)
    idx = 1
    for x1 in x1s, x2 in x2s
        xs[:, idx] = [x1, x2]
        idx += 1
    end
    
    matern52(r) = (1 + sqrt(5) * r / ℓ + 5 * r^2 / (3 * ℓ^2)) * exp(-sqrt(5) * r / ℓ)

    K = Matrix{Float64}(undef, n_total, n_total)
    for i in 1:n_total
        K[i, i] = amplitude^2 + jitter
        for j in (i + 1):n_total
            r = norm(xs[:, i] - xs[:, j])
            val = amplitude^2 * matern52(r)
            K[i, j] = val
            K[j, i] = val
        end
    end

    y = rand(rng, MvNormal(zeros(n_total), Symmetric(K)))

    # lightweight bilinear interpolant
    function f_interp(x::AbstractVector{<:Real})
        xx1 = clamp(x[1], bounds[1][1], bounds[2][1])
        xx2 = clamp(x[2], bounds[1][2], bounds[2][2])
        
        idx1 = searchsortedfirst(x1s, xx1)
        idx2 = searchsortedfirst(x2s, xx2)
        
        if idx1 == 1
            idx1 = 2
        elseif idx1 > length(x1s)
            idx1 = length(x1s)
        end
        
        if idx2 == 1
            idx2 = 2
        elseif idx2 > length(x2s)
            idx2 = length(x2s)
        end
        
        x1_lo, x1_hi = x1s[idx1 - 1], x1s[idx1]
        x2_lo, x2_hi = x2s[idx2 - 1], x2s[idx2]
        
        t1 = (xx1 - x1_lo) / (x1_hi - x1_lo)
        t2 = (xx2 - x2_lo) / (x2_hi - x2_lo)
        
        # Get values at corners
        y_ll = y[(idx2 - 2) * n_points + (idx1 - 1)]
        y_lh = y[(idx2 - 1) * n_points + (idx1 - 1)]
        y_hl = y[(idx2 - 2) * n_points + idx1]
        y_hh = y[(idx2 - 1) * n_points + idx1]
        
        # Bilinear interpolation
        y_interp = (1 - t1) * (1 - t2) * y_ll + 
                   (1 - t1) * t2 * y_lh + 
                   t1 * (1 - t2) * y_hl + 
                   t1 * t2 * y_hh
        
        return y_interp
    end

    return f_interp
end


function tnp_test_2d_multioutput()
    bounds = ((0.0, 0.0), (1.0, 1.0))

    # Sample a 2D objective from a GP prior with 2 output dimensions
    f1_sample = sample_gp_function_2d(; bounds)
    f2_sample = sample_gp_function_2d(; bounds)
    problem = GP2DProblemMultiOutput(f1_sample, f2_sample, bounds)
    sim = simulator(problem)
    f = true_f(problem)

    # Training data (sparse grid)
    n_train = 5
    x1_train = range(bounds[1][1], bounds[2][1], length = n_train)
    x2_train = range(bounds[1][2], bounds[2][2], length = n_train)
    
    X_train = Matrix{Float64}(undef, 2, n_train^2)
    Y_train = Matrix{Float64}(undef, 2, n_train^2)
    
    idx = 1
    for x1 in x1_train, x2 in x2_train
        x = [x1, x2]
        X_train[:, idx] = x
        Y_train[:, idx] = sim(x)
        idx += 1
    end

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
    n_test = 50
    x1_test = range(bounds[1][1], bounds[2][1], length = n_test)
    x2_test = range(bounds[1][2], bounds[2][2], length = n_test)

    # Initialize arrays for both outputs
    Z_true_out1 = Matrix{Float64}(undef, n_test, n_test)
    Z_true_out2 = Matrix{Float64}(undef, n_test, n_test)
    Z_mean_gp_out1 = Matrix{Float64}(undef, n_test, n_test)
    Z_std_gp_out1 = Matrix{Float64}(undef, n_test, n_test)
    Z_mean_gp_out2 = Matrix{Float64}(undef, n_test, n_test)
    Z_std_gp_out2 = Matrix{Float64}(undef, n_test, n_test)
    Z_mean_tnp_out1 = Matrix{Float64}(undef, n_test, n_test)
    Z_std_tnp_out1 = Matrix{Float64}(undef, n_test, n_test)
    Z_mean_tnp_out2 = Matrix{Float64}(undef, n_test, n_test)
    Z_std_tnp_out2 = Matrix{Float64}(undef, n_test, n_test)
    Z_post_mean_gp = Matrix{Float64}(undef, n_test, n_test)
    Z_post_std_gp = Matrix{Float64}(undef, n_test, n_test)
    Z_post_mean_tnp = Matrix{Float64}(undef, n_test, n_test)
    Z_post_std_tnp = Matrix{Float64}(undef, n_test, n_test)
    Z_post_true = Matrix{Float64}(undef, n_test, n_test)

    logpost_true = true_logpost(problem)

    @showprogress desc = "Evaluating" for (i, x1) in enumerate(x1_test), (j, x2) in enumerate(x2_test)
        x = [x1, x2]
        
        f_out = f(x)
        Z_true_out1[j, i] = f_out[1]
        Z_true_out2[j, i] = f_out[2]
        
        m_gp, v_gp = mean_and_var(model_posterior_gp, x)
        Z_mean_gp_out1[j, i] = m_gp[1]
        Z_std_gp_out1[j, i] = sqrt(v_gp[1])
        Z_mean_gp_out2[j, i] = m_gp[2]
        Z_std_gp_out2[j, i] = sqrt(v_gp[2])
        
        m_tnp, v_tnp = mean_and_var(model_posterior_tnp, x)
        Z_mean_tnp_out1[j, i] = m_tnp[1]
        Z_std_tnp_out1[j, i] = sqrt(v_tnp[1])
        Z_mean_tnp_out2[j, i] = m_tnp[2]
        Z_std_tnp_out2[j, i] = sqrt(v_tnp[2])

        Z_post_true[j, i] = exp(logpost_true(x))
        Z_post_mean_gp[j, i] = exp(logpost_mean_gp(x))
        Z_post_std_gp[j, i] = sqrt(exp(logpost_var_gp(x)))
        Z_post_mean_tnp[j, i] = exp(logpost_mean_tnp(x))
        Z_post_std_tnp[j, i] = sqrt(exp(logpost_var_tnp(x)))
    end

    # Plotting - 3 rows (means for output 1, means for output 2, posteriors)
    fig = Figure(size = (1400, 1200))

    z_obs = likelihood(problem).z_obs

    # Row 1: Output 1 means
    ax_true1 = Axis(fig[1, 1]; title = "True (output 1)", xlabel = "x₁", ylabel = "x₂")
    hm_true1 = heatmap!(ax_true1, x1_test, x2_test, Z_true_out1)
    scatter!(ax_true1, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[1, 2], hm_true1)

    ax1 = Axis(fig[1, 3]; title = "GP mean (output 1)", xlabel = "x₁", ylabel = "x₂")
    hm1 = heatmap!(ax1, x1_test, x2_test, Z_mean_gp_out1)
    scatter!(ax1, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[1, 4], hm1)

    ax2 = Axis(fig[1, 5]; title = "TNP mean (output 1)", xlabel = "x₁", ylabel = "x₂")
    hm2 = heatmap!(ax2, x1_test, x2_test, Z_mean_tnp_out1)
    scatter!(ax2, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[1, 6], hm2)

    # Row 2: Output 2 means
    ax_true2 = Axis(fig[2, 1]; title = "True (output 2)", xlabel = "x₁", ylabel = "x₂")
    hm_true2 = heatmap!(ax_true2, x1_test, x2_test, Z_true_out2)
    scatter!(ax_true2, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[2, 2], hm_true2)

    ax3 = Axis(fig[2, 3]; title = "GP mean (output 2)", xlabel = "x₁", ylabel = "x₂")
    hm3 = heatmap!(ax3, x1_test, x2_test, Z_mean_gp_out2)
    scatter!(ax3, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[2, 4], hm3)

    ax4 = Axis(fig[2, 5]; title = "TNP mean (output 2)", xlabel = "x₁", ylabel = "x₂")
    hm4 = heatmap!(ax4, x1_test, x2_test, Z_mean_tnp_out2)
    scatter!(ax4, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[2, 6], hm4)

    # Row 3: Posteriors
    ax_post_true = Axis(fig[3, 1]; title = "True posterior", xlabel = "x₁", ylabel = "x₂")
    hm_post_true = heatmap!(ax_post_true, x1_test, x2_test, Z_post_true)
    scatter!(ax_post_true, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[3, 2], hm_post_true)

    ax_post_gp = Axis(fig[3, 3]; title = "GP posterior", xlabel = "x₁", ylabel = "x₂")
    hm_post_gp = heatmap!(ax_post_gp, x1_test, x2_test, Z_post_mean_gp)
    scatter!(ax_post_gp, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[3, 4], hm_post_gp)

    ax_post_tnp = Axis(fig[3, 5]; title = "TNP posterior", xlabel = "x₁", ylabel = "x₂")
    hm_post_tnp = heatmap!(ax_post_tnp, x1_test, x2_test, Z_post_mean_tnp)
    scatter!(ax_post_tnp, X_train[1,:], X_train[2,:]; color = :white, markersize = 6, strokewidth = 1, strokecolor = :black)
    Colorbar(fig[3, 6], hm_post_tnp)

    save("plots/test_2d_multioutput.png", fig)
    fig
end
