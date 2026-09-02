"""
    example_asymptotics_analysis.jl

Example script demonstrating how to use the asymptotics_analytics module
to load, analyze, and visualize convergence behavior across dimensions.

## Visualization Guide

The refactored analytics produces several key visualizations:

**Color scheme throughout:**
- **Blue**: Gradient-enhanced method (grads-warm) — typically achieves better (lower) convergence curve
- **Orange**: Plain method (standard-warm)
- Lower TV/error = better

### Main Plot: slope_comparison.png
Shows slope ratios βg/β0 (gradient GP convergence rate / plain GP rate) as a function of dimension:

**Left panel - Slope Ratio Comparison:**
- **Measured (black points with error bars)**: Empirical slope ratios from your experiments
  - Computed from multiple runs per dimension
  - Error bars show ±1σ uncertainty across runs
  
- **Asymptotic (blue dashed)**: Hermite order effect only
  - Formula: 1 + 2d / [(2ν+d)(2ν+2d+2)]
  - Represents the theoretical slope improvement from derivative information in the limit n → ∞
  - Typically small (~5% for Matérn-5/2), nearly constant across d
  - Applies when both methods are well past the phase transition
  
- **Pre-asymptotic (red dashed)**: Cost advantage factor (1+d)/r_c
  - Represents the iteration count advantage from gradient observations
  - For r_c=2.5 and d=10: factor of 4.4
  - Dominates in practical regimes where n is small relative to critical sample size
  
- **Unified (purple solid)**: Full prediction combining both effects
  - Formula: (1+d)/r_c × Hermite_factor
  - Expected most accurate for practical experimental regimes

**Right panel - Critical Sample Size:**
- Shows N_crit ≈ d^(d/ν) where the phase transition occurs
- For d=2: ~2.8; d=5: ~97.7; d=10: ~10,000
- Gradient GP reaches learning phase faster because it has (1+d)× more effective observations
- Orange line indicates typical budget (e.g., 100 evaluations)
- When n < N_crit, both methods are underfitting; gradient advantage is maximized
- When n > N_crit, both are learning; advantage shrinks toward Hermite factor

### Theory Breakdown Plot: theory_breakdown.png
Detailed analysis of theoretical contributions:

**Left panel - Component Factors:**
- Shows individual multipliers on log scale
- Hermite factor (≈1.05): small asymptotic advantage
- Cost factor (1+d)/r_c: grows linearly with dimension
- Product (unified): their combination

**Right panel - Contribution Breakdown:**
- Bars show how much each component contributes to the slope improvement
- Illustrates the dominance of cost factor at low d vs. Hermite at high d
- Reveals the transition from pre- to asymptotic regime

### Separate Problem Type Comparison: slope_comparison_by_problem.png
Shows measured slope ratios separately for each problem type (SimpleProblem and ABProblem):

- **One subplot per problem type** (e.g., SimpleProblem | ABProblem)
- Each subplot shows the same elements as the main plot:
  - Measured slopes with uncertainty bands
  - Theoretical predictions (asymptotic, pre-asymptotic, unified)
- Allows direct visual comparison of how the methods' relative performance differs between problem types
- Useful for diagnosing whether the gradient advantage is problem-dependent

### Cost-Adjusted Slope Comparison: slope_comparison_cost_adjusted.png

**Understanding why gradient method wins even when slope ratio < 1:**

Looking at your convergence plots:
- **Blue = gradient method (grads-warm)**: achieves **lower TV = better results**
- **Orange = plain method (standard-warm)**

Yet our slope ratio plot shows β_g/β_0 < 1 in some dimensions. This seems paradoxical—how can 
the gradient method win if it has a lower slope?

**The answer: Intercept dominates over slope.**

**Left panel - Measured Slope Ratios:**
- Shows βg/β0 per simulation count (not per observation)
- Values < 1 mean gradient converges *slower per simulation* than plain
- This is NOT bad—it means gradient already has low error and additional iterations help less

**Right panel - Why Gradient Still Achieves Best Results:**
- **Blue line (1+d)**: Gradient gets (1+d) observations per simulation vs 1 for plain
  - At d=2: 3× more observations per call
  - At d=10: 11× more observations per call
- **Red line (1+d)/r_c**: After accounting for gradient's computational cost (r_c≈2.5)
  - At d=2: ~1.2× effective advantage per simulation cost
  - At d=10: ~4.4× effective advantage per simulation cost

**The mechanism:**
1. Gradient method starts with **much lower error** (better intercept) due to (1+d) observations
2. Plain method may have slightly steeper slope (more room for improvement)
3. But gradient's lead is so large that it reaches target error with fewer total simulations
4. At high dimensions, gradient's information advantage (1+d) far outweighs any slope disadvantage

This is exactly why gradients work best for **high-dimensional problems** where (1+d) is large!

## Physical Interpretation

The pre-asymptotic regime (most practical applications) shows why gradient observations help:
1. **Iteration advantage**: Each gradient eval gives (1+d) scalar observations
2. **Phase transition**: Gradient GP reaches the learning phase in fewer evaluations
   - Dimension 5: ~10× fewer evaluations needed
   - Dimension 10: ~50× fewer evaluations needed
3. **Cost constraint**: Speedup only realized if r_c < 1 + d
   - Automatic differentiation: r_c ≈ 2-3, beneficial for all d ≥ 2
   - Finite differences: r_c ≈ 1+d, speedup cancels out

In the asymptotic regime (n >> d^(d/ν)), both methods reach learning phase and slopes
become nearly identical. The measured slope differences should approach the Hermite factor (~5%).

## Usage

```julia
# Run basic analysis (automatic plotting)
analysis = example_basic_analysis()

# Run with manual control and custom visualizations
# analysis = example_detailed_analysis()

# Access raw data structure:
# analysis.dimensions → [2, 4, 6, 8, 10]
# analysis.results[d]["plain"] → [fit1, fit2, ...] (list of PowerLawFit objects)
# analysis.results[d]["grad"]  → [fit1, fit2, ...] for gradient method
```

"""

using JLD2
using Glob
using Statistics

# Include the analytics module
include("asymptotics_analytics.jl")

function example_basic_analysis()
    """
    Run basic analysis on experiment data with default settings.
    
    This will:
    1. Load all "standard-warm" and "grads-warm" run data from the data directory
    2. Fit power laws to the convergence curves
    3. Print summary statistics and theory comparisons
    4. Generate comprehensive plots comparing slopes across dimensions with:
       - Measured slope ratios (with error bars)
       - Asymptotic theoretical predictions (Hermite order effect)
       - Pre-asymptotic theoretical predictions (phase transition / cost advantage)
       - Unified predictions combining both effects
       - Critical iteration counts showing phase transition points
    """
    
    # Specify the data directory (should contain subdirectories like MultidimProblem{ABProblem}1, etc.)
    data_dir = joinpath(@__DIR__, "..", "data-convergence4")

    if !isdir(data_dir)
        error("Data directory not found: $data_dir")
    end

    # Run the main analysis
    # Parameters:
    #   r_c: relative cost ratio (gradient eval cost / value-only eval cost)
    #        typical values: ~2-3 for automatic differentiation
    #   ν: Matérn smoothness parameter (ν=2.5 = Matérn-5/2, commonly used)
    analysis = main_analysis(data_dir; 
                            plot_output_dir="plots/convergence",
                            r_c=2.5,           # relative cost of gradients vs value
                            ν=2.5,             # Matérn-5/2 kernel (ν=2.5)
                            verbose=true)      # print progress
    
    return analysis
end

function example_detailed_analysis()
    """
    Run detailed analysis with manual data loading and custom processing.
    
    This provides more control over options and allows custom post-processing.
    """
    
    data_dir = joinpath(@__DIR__, "..", "data-convergence4")

    # Step 1: Load data with custom options
    println("\n" * "="^80)
    println("Step 1: Loading experiment data")
    println("="^80)
    
    data = load_experiment_data_by_run(data_dir;
                                       methods=["standard-warm", "grads-warm"],
                                       expected_runs=5,
                                       rename_methods=true)  # Rename to "plain"/"grad"
    
    # Print loaded data structure
    println("\nData structure:")
    for dim in sort(collect(keys(data)))
        println("  Dimension $dim:")
        for method in collect(keys(data[dim]))
            n_runs = length(data[dim][method])
            println("    - $method: $n_runs runs")
        end
    end
    
    # Step 2: Fit power laws
    println("\n" * "="^80)
    println("Step 2: Fitting power laws")
    println("="^80)
    
    analysis = analyze_multiple_runs(data; verbose=true)
    
    # Step 3: Print results
    println("\n" * "="^80)
    println("Step 3: Results Summary")
    println("="^80)
    
    summary_table(analysis)
    theory_comparison_table(analysis; r_c=2.5, ν=2.5)
    
    # Step 4: Custom visualization
    println("\n" * "="^80)
    println("Step 4: Generating detailed visualizations")
    println("="^80)
    
    # Create plots directory if needed
    if !isdir("plots/convergence")
        mkpath("plots/convergence")
    end
    
    # Generate main slope comparison plot
    println("\nCreating slope comparison plot...")
    p_slopes = plot_slope_comparison(analysis; r_c=2.5, ν=2.5)
    save("plots/convergence/slope_comparison_detailed.png", p_slopes)
    println("  → slope_comparison_detailed.png")
    
    # Generate theory breakdown plot
    println("Creating theory breakdown plot...")
    p_theory = plot_detailed_theory_analysis(analysis; r_c=2.5, ν=2.5)
    save("plots/convergence/theory_breakdown_detailed.png", p_theory)
    println("  → theory_breakdown_detailed.png")
    
    # Step 5: Custom analysis per dimension
    println("\n" * "="^80)
    println("Step 5: Custom analysis per dimension")
    println("="^80)
    
    for d in analysis.dimensions
        if haskey(analysis.results[d], "plain") && haskey(analysis.results[d], "grad")
            fits_plain = analysis.results[d]["plain"]
            fits_grad = analysis.results[d]["grad"]
            
            β_plain_vals = [f.β for f in fits_plain]
            β_grad_vals = [f.β for f in fits_grad]
            
            β_plain_mean = mean(β_plain_vals)
            β_grad_mean = mean(β_grad_vals)
            
            slope_ratio = β_grad_mean / β_plain_mean
            
            # Theoretical predictions
            hermite_pred = theoretical_slope_ratio(d; ν=2.5)
            cost_pred = theoretical_cost_advantage(d; r_c=2.5)
            unified_pred = theoretical_unified_slope_ratio(d; r_c=2.5, ν=2.5, regime=:pre_asymptotic)
            n_crit = critical_iteration_count(d; ν=2.5)
            
            println("\nDimension $d:")
            println("  Measured slope ratio βg/β0 = $(round(slope_ratio; digits=4))")
            println("  Plain method (standard-warm):")
            println("    - Mean slope β = $(round(β_plain_mean; digits=4)) ± $(round(std(β_plain_vals); digits=4))")
            println("  Grad method (grads-warm):")
            println("    - Mean slope β = $(round(β_grad_mean; digits=4)) ± $(round(std(β_grad_vals); digits=4))")
            println("  Theoretical predictions:")
            println("    - Asymptotic (Hermite) = $(round(hermite_pred; digits=4))")
            println("    - Pre-asymptotic (cost) = $(round(cost_pred; digits=4))")
            println("    - Unified (both effects) = $(round(unified_pred; digits=4))")
            println("  Critical sample size N_crit ≈ $(n_crit)")
            
            # Get confidence interval for one of the fits
            if !isempty(fits_grad)
                fit = fits_grad[1]
                ci_lower, ci_upper = confidence_interval_slope(fit; α_level=0.05)
                println("  Grad slope 95% CI for first run: [$(round(ci_lower; digits=4)), $(round(ci_upper; digits=4))]")
            end
        end
    end
    
    return analysis
end

function example_bootstrap_analysis()
    """
    Example of using bootstrap to estimate slope uncertainty.
    """
    
    # Generate example scores (exponential decay - typical convergence behavior)
    n_iterations = 100
    scores = [0.5 * exp(-0.02 * i) + 0.01 * randn() for i in 1:n_iterations]
    
    println("\n" * "="^80)
    println("Bootstrap Uncertainty Estimation Example")
    println("="^80)
    
    # Fit single power law
    fit = fit_powerlaw(scores; verbose=true)
    
    # Estimate uncertainty via bootstrap
    β_mean, β_std, (ci_lower, ci_upper), slopes = bootstrap_slope(scores; n_bootstrap=1000, verbose=false)
    
    println("\nBootstrap Results (1000 replicates):")
    println("  Mean slope (bootstrap) = $β_mean")
    println("  Std dev (bootstrap) = $β_std")
    println("  95% CI (bootstrap) = [$ci_lower, $ci_upper]")
    println("  Direct fit slope = $(fit.β)")
    
    return fit, (β_mean, β_std, (ci_lower, ci_upper))
end

# Run example (uncomment the one you want)
if abspath(PROGRAM_FILE) == @__FILE__
    try
        # Run the basic analysis (recommended for first-time use)
        analysis = example_basic_analysis()
        
        # Alternatively, run detailed analysis for more control and insight:
        # analysis = example_detailed_analysis()
        
        # Or demonstrate bootstrap:
        # fit, bootstrap_results = example_bootstrap_analysis()
        
        println("\n" * "="^80)
        println("✓ Analysis complete!")
        println("="^80)
        println("\nGenerated plots:")
        println("  • plots/slope_comparison.png (main visualization with critical iterations)")
        println("  • plots/slope_comparison_cost_adjusted.png (explains low-d paradox)")
        println("  • plots/theory_breakdown.png (theory component breakdown)")
        println("  • plots/slope_comparison_by_problem.png (separate comparison by problem type)")
        println("  • plots/convergence_d*_*_run*.png (individual convergence curves)")
        println("\nSee the documentation at the top of example_asymptotics_analysis.jl")
        println("for interpretation of the plots.")
        
    catch e
        println("Error running example:")
        println(e)
        println("\nMake sure:")
        println("  1. You are in the src directory")
        println("  2. The data directory exists with experiment results")
        println("  3. Data files are stored as: data-convergence4/{problem}/{method}_{run_idx}_TVmetric.jld2")
    end
end
