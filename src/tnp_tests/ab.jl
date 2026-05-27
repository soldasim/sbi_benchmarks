include("../main.jl")

# estimator: "mean" or "approx"
function tnp_test(; estimator::String="mean")
    # Problem setup
    problem = ABProblem()
    sim = simulator(problem)
    bounds = domain(problem).bounds
    
    # Generate training data on a sparse grid
    n_train = 5  # points per dimension
    x1_train = range(bounds[1][1], bounds[2][1], length=n_train)
    x2_train = range(bounds[1][2], bounds[2][2], length=n_train)
    
    X_train = Matrix{Float64}(undef, 2, n_train^2)
    Y_train = Matrix{Float64}(undef, 1, n_train^2)
    
    idx = 1
    for x1 in x1_train, x2 in x2_train
        x = [x1, x2]
        X_train[:, idx] = x
        Y_train[:, idx] = sim(x)
        idx += 1
    end
    
    # Create training data
    data = BOSS.ExperimentData(X_train, Y_train)
    
    # Initialize both models
    SCALE = 5.
    SCALE_OUT = 20.

    tnp_model = TransformedModel(;
        base_model = TNP(),
        input_transform = InputTransform(x -> x ./ SCALE),
        output_transform = SlicedOutputTransform(
            [(y_, std_) -> (y_, std_) .* SCALE_OUT],
            [y -> y / SCALE_OUT],
        ),
    )
    gp_model = TransformedModel(;
        base_model = GaussianProcess(;
            mean = prior_mean(problem),
            kernel = BOSS.Matern52Kernel(),
            # lengthscale_priors = get_lengthscale_priors(problem),
            lengthscale_priors = get_lengthscale_priors(bounds ./ SCALE, y_dim(problem)),
            amplitude_priors = get_amplitude_priors(problem),
            noise_std_priors = get_noise_std_priors(problem),
        ),
        input_transform = InputTransform(x -> x ./ SCALE),
        output_transform = SlicedOutputTransform(
            [(y_, std_) -> (y_, std_) .* SCALE_OUT],
            [y -> y / SCALE_OUT],
        ),
    )
    
    # Generate test grid for plotting
    n_test = 50
    x1_test = range(bounds[1][1], bounds[2][1], length=n_test)
    x2_test = range(bounds[1][2], bounds[2][2], length=n_test)
    
    # Evaluate true simulator on grid
    Z_true = Matrix{Float64}(undef, n_test, n_test)
    
    for (i, x1) in enumerate(x1_test), (j, x2) in enumerate(x2_test)
        x = [x1, x2]
        Z_true[j, i] = sim(x)[1]
    end
    
    # Evaluate posteriors on grid
    logpost_true = true_logpost(problem)
    
    # Construct BosipProblems for both models
    acquisition = LogMaxVar()  # placeholder, not used for posterior evaluation
    
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
    
    # Fit the model parameters
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 24,
        parallel = false,
        rhoend = 1e-4,
    )
    
    println("Fitting GP model...")
    BOSIP.estimate_parameters!(bosip_gp, model_fitter)
    
    println("Fitting TNP model...")
    BOSIP.estimate_parameters!(bosip_tnp, model_fitter)
    
    # Obtain ModelPosteriors
    model_posterior_gp = BOSS.model_posterior(bosip_gp.problem)
    model_posterior_tnp = BOSS.model_posterior(bosip_tnp.problem)
    
    # Select estimator
    if estimator == "mean"
        logpost_gp = BOSIP.log_posterior_mean(bosip_gp)
        logpost_tnp = BOSIP.log_posterior_mean(bosip_tnp)
    elseif estimator == "approx"
        logpost_gp = BOSIP.log_approx_posterior(bosip_gp)
        logpost_tnp = BOSIP.log_approx_posterior(bosip_tnp)
    else
        error("Unknown estimator: $estimator")
    end

    Z_post_true = Matrix{Float64}(undef, n_test, n_test)
    Z_model_gp = Matrix{Float64}(undef, n_test, n_test)
    Z_std_gp = Matrix{Float64}(undef, n_test, n_test)
    Z_post_gp = Matrix{Float64}(undef, n_test, n_test)
    Z_model_tnp = Matrix{Float64}(undef, n_test, n_test)
    Z_std_tnp = Matrix{Float64}(undef, n_test, n_test)
    Z_post_tnp = Matrix{Float64}(undef, n_test, n_test)
    
    @showprogress desc="Computing posterior" for (i, x1) in enumerate(x1_test), (j, x2) in enumerate(x2_test)
        x = [x1, x2]
        
        # GP approximation
        m_gp, v_gp = mean_and_var(model_posterior_gp, x)
        Z_model_gp[j, i] = m_gp[1]
        Z_std_gp[j, i] = sqrt(v_gp[1])
        
        # TNP approximation
        m_tnp, v_tnp = mean_and_var(model_posterior_tnp, x)
        Z_model_tnp[j, i] = m_tnp[1]
        Z_std_tnp[j, i] = sqrt(v_tnp[1])
        
        # True posterior
        Z_post_true[j, i] = exp(logpost_true(x))
        
        # Model-based posteriors
        Z_post_gp[j, i] = exp(logpost_gp(x))
        Z_post_tnp[j, i] = exp(logpost_tnp(x))
    end
    
    # Create plots (3x3 layout)
    fig = Figure(size=(1400, 1200))
    
    # Column 1: True
    ax1 = Axis(fig[1, 1]; 
        title="True Simulator (a × b)",
        xlabel="a",
        ylabel="b"
    )
    hm1 = heatmap!(ax1, x1_test, x2_test, Z_true)
    scatter!(ax1, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[1, 2], hm1)
    
    # Row 2: Model std (column 1 is empty)
    ax2a = Axis(fig[2, 3];
        title="GP Std",
        xlabel="a", 
        ylabel="b"
    )
    hm2a = heatmap!(ax2a, x1_test, x2_test, Z_std_gp)
    scatter!(ax2a, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[2, 4], hm2a)
    
    ax2b = Axis(fig[2, 5];
        title="TNP Std",
        xlabel="a", 
        ylabel="b"
    )
    hm2b = heatmap!(ax2b, x1_test, x2_test, Z_std_tnp)
    scatter!(ax2b, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[2, 6], hm2b)
    
    # Row 3: Model mean
    ax3 = Axis(fig[3, 1];
        title="True Posterior",
        xlabel="a",
        ylabel="b"
    )
    hm3 = heatmap!(ax3, x1_test, x2_test, Z_post_true)
    scatter!(ax3, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[3, 2], hm3)
    
    # Column 2: GP
    ax4 = Axis(fig[1, 3];
        title="GP",
        xlabel="a", 
        ylabel="b"
    )
    hm4 = heatmap!(ax4, x1_test, x2_test, Z_model_gp)
    scatter!(ax4, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[1, 4], hm4)
    
    ax5 = Axis(fig[3, 3];
        title="GP Posterior ($estimator)",
        xlabel="a",
        ylabel="b"
    )
    hm5 = heatmap!(ax5, x1_test, x2_test, Z_post_gp)
    scatter!(ax5, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[3, 4], hm5)
    
    # Column 3: TNP
    ax6 = Axis(fig[1, 5];
        title="TNP",
        xlabel="a", 
        ylabel="b"
    )
    hm6 = heatmap!(ax6, x1_test, x2_test, Z_model_tnp)
    scatter!(ax6, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[1, 6], hm6)
    
    ax7 = Axis(fig[3, 5];
        title="TNP Posterior ($estimator)",
        xlabel="a",
        ylabel="b"
    )
    hm7 = heatmap!(ax7, x1_test, x2_test, Z_post_tnp)
    scatter!(ax7, X_train[1,:], X_train[2,:]; color=:white, markersize=8, strokewidth=2, strokecolor=:black)
    Colorbar(fig[3, 6], hm7)
    
    save("plots/test.png", fig)
    return fig
end
