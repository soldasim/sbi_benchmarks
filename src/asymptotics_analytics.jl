"""
    asymptotics_analysis.jl

Tools for fitting power laws to convergence data and testing asymptotic theory predictions.

Power law model: log(score) = α + β * log(n)
where n ∈ {1, 2, ..., N} are iteration indices.

Usage:
    scores = [0.5, 0.3, 0.2, ...]  # vector of N scores
    fit = fit_powerlaw(scores)
    slope = fit.β
    ci = confidence_interval_slope(fit)
"""

using Statistics
using LinearAlgebra
using StatsBase
using Distributions
using JLD2
using Glob
using Printf
using CairoMakie

import StatsModels: @formula, modelmatrix, coefnames
import GLM: lm, coef, vcov

# ============================================================================
# Core Power Law Fitting
# ============================================================================

struct PowerLawFit
    """Fitted power law: log(score) = α + β * log(n)"""
    α::Float64                          # intercept
    β::Float64                          # slope
    n_samples::Int                      # number of iterations (length of score vector)
    log_n::Vector{Float64}              # log-transformed iteration indices
    log_scores::Vector{Float64}         # log-transformed scores
    residuals::Vector{Float64}          # residuals from fit
    σ::Float64                          # estimated standard error
    dof::Int                            # degrees of freedom (n - 2)
    model                               # underlying GLM regression object
end

"""
    fit_powerlaw(scores::Vector; verbose::Bool=false)

Fit power law to scores via log-log regression.

Args:
    scores: Vector of length N with score values (should be positive)
    verbose: Print fit details

Returns:
    PowerLawFit object with slope β and intercept α
"""
function fit_powerlaw(scores::Vector; verbose::Bool=false)
    n = length(scores)
    
    # Check validity
    if any(scores .<= 0)
        error("All scores must be positive for log-log fit")
    end
    
    # Prepare data
    log_n = log.(1:n)
    log_scores = log.(scores)
    
    # Fit linear regression: log(scores) ~ log(n)
    X = hcat(ones(n), log_n)  # design matrix [1, log(n)]
    β_est = X \ log_scores    # least squares solution
    
    α, β = β_est[1], β_est[2]
    
    # Predictions and residuals
    fitted = α .+ β .* log_n
    residuals = log_scores .- fitted
    
    # Standard error
    σ_squared = sum(residuals .^ 2) / (n - 2)
    σ = sqrt(σ_squared)
    
    # Create model wrapper for GLM compatibility (for confidence intervals)
    # We'll refit with GLM to get proper CI support
    df = (; log_n = log_n, log_scores = log_scores)
    model = lm(@formula(log_scores ~ log_n), df)
    
    if verbose
        println("Power Law Fit Results")
        println("=" ^ 50)
        println("Model: log(score) = α + β * log(n)")
        println("Fitted values:")
        println("  α (intercept) = $(round(α; digits=6))")
        println("  β (slope)     = $(round(β; digits=6))")
        println("  σ (std err)   = $(round(σ; digits=6))")
        println("  n (samples)   = $n")
        println("=" ^ 50)
    end
    
    return PowerLawFit(α, β, n, log_n, log_scores, residuals, σ, n-2, model)
end

# ============================================================================
# Confidence Intervals & Uncertainty
# ============================================================================

"""
    confidence_interval_slope(fit::PowerLawFit; α_level::Float64=0.05)

Compute confidence interval for the slope β.

Args:
    fit: PowerLawFit object
    α_level: significance level (default 0.05 for 95% CI)

Returns:
    (β_lower, β_upper): bounds of the confidence interval
"""
function confidence_interval_slope(fit::PowerLawFit; α_level::Float64=0.05)
    # Use GLM-computed standard errors
    vcov_matrix = vcov(fit.model)
    σ_β = sqrt(vcov_matrix[2, 2])  # std error of β (second coefficient)
    
    # t-critical value
    t_crit = quantile(TDist(fit.dof), 1 - α_level/2)
    
    β_lower = fit.β - t_crit * σ_β
    β_upper = fit.β + t_crit * σ_β
    
    return (β_lower, β_upper)
end

"""
    bootstrap_slope(scores::Vector; n_bootstrap::Int=1000, verbose::Bool=false)

Estimate slope uncertainty via bootstrap resampling.

Args:
    scores: Score vector
    n_bootstrap: Number of bootstrap replicates
    verbose: Print progress

Returns:
    (β_mean, β_std, β_percentile_ci): slope estimate & CI from bootstrap
"""
function bootstrap_slope(scores::Vector; n_bootstrap::Int=1000, verbose::Bool=false)
    slopes = Float64[]
    
    for i in 1:n_bootstrap
        # Resample with replacement
        idx = rand(1:length(scores), length(scores))
        scores_boot = scores[idx]
        
        fit = fit_powerlaw(scores_boot; verbose=false)
        push!(slopes, fit.β)
        
        if verbose && i % 100 == 0
            println("Bootstrap iteration $i / $n_bootstrap")
        end
    end
    
    β_mean = mean(slopes)
    β_std = std(slopes)
    β_lower = quantile(slopes, 0.025)
    β_upper = quantile(slopes, 0.975)
    
    return (β_mean, β_std, (β_lower, β_upper), slopes)
end

# ============================================================================
# Theory Predictions
# ============================================================================

"""
    theoretical_slope_ratio(d::Int; ν::Float64=2.5)

Predict slope ratio β_grad / β_plain from Hermite order theory (§5).

For Matérn-ν GP in d dimensions:
Slope ratio = 1 + 2d / [(2ν + d)(2ν + 2d + 2)]

Args:
    d: Dimension
    ν: Matérn smoothness parameter (default 2.5 for Matérn-5/2)

Returns:
    Predicted slope ratio
"""
function theoretical_slope_ratio(d::Int; ν::Float64=2.5)
    numerator = 2 * d
    denominator = (2*ν + d) * (2*ν + 2*d + 2)
    return 1 + numerator / denominator
end

"""
    theoretical_cost_advantage(d::Int; r_c::Float64=2.5)

Predict cost advantage (1+d)/r_c from phase transition theory (§4, §6).

This is the factor by which grad-BOSIP reaches a given error faster
than plain BOSIP (in terms of number of simulator calls).

Args:
    d: Dimension
    r_c: Relative cost of gradient evaluation vs. value only (default 2.5 for AD)

Returns:
    Predicted cost advantage factor
"""
function theoretical_cost_advantage(d::Int; r_c::Float64=2.5)
    return (1 + d) / r_c
end

"""
    theoretical_unified_slope_ratio(d::Int; r_c::Float64=2.5, ν::Float64=2.5, regime::Symbol=:pre_asymptotic)

Predict slope ratio from theory (Hermite order correction).

**IMPORTANT:** The slope ratio is determined by the Hermite order (approximation order 
improvement from derivative observations) and is independent of the cost factor (1+d)/r_c.

The cost factor (1+d)/r_c manifests as a VERTICAL OFFSET on the log-log plot 
(number of calls saved), not as a change in slope.

Slope ratio ≈ 1 + 2d/((2ν+d)(2ν+2d+2)) ≈ 1.05 for Matérn-5/2, regardless of regime or d.

Args:
    d: Dimension
    r_c: Relative cost (used only for documentation, does not affect slope ratio)
    ν: Matérn parameter (default 2.5)
    regime: :pre_asymptotic or :asymptotic (both give same slope ratio; kept for API compatibility)

Returns:
    Predicted slope ratio β_g / β_0 (always dominated by Hermite order ≈ 1.05)
"""
function theoretical_unified_slope_ratio(d::Int; r_c::Float64=2.5, ν::Float64=2.5, 
                                        regime::Symbol=:pre_asymptotic)
    # Slope ratio is determined purely by Hermite order, independent of cost factor
    return theoretical_slope_ratio(d; ν=ν)
end

# ============================================================================
# Cost Advantage & Slope Analysis
# ============================================================================

"""
    cost_advantage_at_error(scores_plain::Vector, scores_grad::Vector, target_error::Float64)

Estimate cost advantage by counting calls to reach a target error.

Args:
    scores_plain: Score vector for plain BOSIP
    scores_grad: Score vector for grad-BOSIP
    target_error: Target TV distance

Returns:
    n_plain / n_grad: ratio of calls to reach target_error

Raises error if target_error not reached by both methods.
"""
function cost_advantage_at_error(scores_plain::Vector, scores_grad::Vector, 
                                 target_error::Float64)
    idx_plain = findfirst(scores_plain .<= target_error)
    idx_grad = findfirst(scores_grad .<= target_error)
    
    if isnothing(idx_plain) || isnothing(idx_grad)
        error("Target error $target_error not reached by both methods")
    end
    
    return idx_plain / idx_grad
end

"""
    cost_advantage_over_range(scores_plain::Vector, scores_grad::Vector; quantiles::Vector{Float64}=[0.1, 0.2, 0.5])

Estimate cost advantage at multiple error levels (quantiles of final scores).

Args:
    scores_plain: Score vector for plain BOSIP
    scores_grad: Score vector for grad-BOSIP
    quantiles: Error thresholds as quantiles of grad-BOSIP final score

Returns:
    Dict mapping quantile → cost advantage
"""
function cost_advantage_over_range(scores_plain::Vector, scores_grad::Vector;
                                   quantiles::Vector{Float64}=[0.1, 0.2, 0.5])
    min_error = minimum(scores_grad)
    advantages = Dict()
    
    for q in quantiles
        target = min_error + q * (maximum(scores_grad) - min_error)
        try
            adv = cost_advantage_at_error(scores_plain, scores_grad, target)
            advantages[q] = adv
        catch
            advantages[q] = NaN
        end
    end
    
    return advantages
end

# ============================================================================
# Multi-Run Analysis
# ============================================================================

struct MultiRunAnalysis
    """Container for analysis across multiple runs and dimensions"""
    dimensions::Vector{Int}
    results::Dict  # dim => {method => [fit1, fit2, ...]}
    problem_types::Vector{String}  # List of unique problem types (e.g., ["ABProblem", "SimpleProblem"])
    results_by_problem::Dict  # problem_type => {dim => {method => [fit1, fit2, ...]}}
end

"""
    organize_data_by_problem_type(data_dir::String)::Tuple{Dict, Dict}

Organize loaded data by problem type.

Returns:
    (data_by_problem, problem_types) where:
    - data_by_problem[problem_type][dim] = {method => [scores_run1, ...]}
    - problem_types = ["ABProblem", "SimpleProblem", ...] (sorted)
"""
function organize_data_by_problem_type(data_dir::String)
    data_by_problem = Dict{String, Dict}()
    problem_types_set = Set{String}()
    
    try
        problem_dirs = readdir(data_dir; join=true)
    catch e
        error("Failed to read data directory $data_dir: $e")
    end
    
    problem_dirs = filter(isdir, problem_dirs)
    problem_dirs = sort(problem_dirs)
    
    for problem_dir in problem_dirs
        problem_type = extract_problem_type_from_path(problem_dir)
        if problem_type == "Unknown"
            continue
        end
        
        push!(problem_types_set, problem_type)
        
        if !haskey(data_by_problem, problem_type)
            data_by_problem[problem_type] = Dict()
        end
    end
    
    problem_types = sort(collect(problem_types_set))
    return data_by_problem, problem_types
end

"""
    analyze_multiple_runs(data::Dict; verbose::Bool=false, data_dir::String="")

Fit power laws across multiple dimensions and methods.

Args:
    data: Dict with structure: dim => {method => [scores_run1, scores_run2, ...]}
          Example: 2 => {"standard-warm" => [[...], [...]], "grads-warm" => [[...], [...]]}
    verbose: Print details
    data_dir: Optional data directory path to extract problem type information

Returns:
    MultiRunAnalysis object with per-dimension, per-method results
"""
function analyze_multiple_runs(data::Dict; verbose::Bool=false, data_dir::String="")
    results = Dict()
    results_by_problem = Dict()
    problem_types = String[]
    
    # First pass: fit data aggregated across all problem types
    for d in sort(collect(keys(data)))
        results[d] = Dict()
        for method in keys(data[d])
            if verbose
                println("\nDimension $d, Method $method")
            end
            results[d][method] = []
            for (run_idx, scores) in enumerate(data[d][method])
                fit = fit_powerlaw(scores; verbose=verbose && run_idx == 1)
                push!(results[d][method], fit)
            end
        end
    end
    
    # Second pass: if data_dir is provided, load and fit data separately for each problem type
    if !isempty(data_dir) && isdir(data_dir)
        _, problem_types = organize_data_by_problem_type(data_dir)
        
        if verbose && !isempty(problem_types)
            println("\nLoading data for $(length(problem_types)) problem type(s): $problem_types")
        end
        
        for problem_type in problem_types
            results_by_problem[problem_type] = Dict()
            
            # Load data for this specific problem type only
            data_for_problem = load_experiment_data_by_run_filtered(data_dir, problem_type)
            
            if verbose
                dims_for_problem = collect(keys(data_for_problem))
                println("  Problem type '$problem_type': found $(length(dims_for_problem)) dimension(s)")
            end
            
            if !isempty(data_for_problem)
                # Fit data for this problem type
                for d in sort(collect(keys(data_for_problem)))
                    results_by_problem[problem_type][d] = Dict()
                    for method in keys(data_for_problem[d])
                        results_by_problem[problem_type][d][method] = []
                        for (run_idx, scores) in enumerate(data_for_problem[d][method])
                            fit = fit_powerlaw(scores; verbose=false)
                            push!(results_by_problem[problem_type][d][method], fit)
                        end
                    end
                end
            end
        end
    end
    
    return MultiRunAnalysis(sort(collect(keys(data))), results, problem_types, results_by_problem)
end

"""
    summary_table(analysis::MultiRunAnalysis; α_level::Float64=0.05)

Print summary table of slopes with confidence intervals.

Args:
    analysis: MultiRunAnalysis object
    α_level: Significance level for confidence intervals
"""
function summary_table(analysis::MultiRunAnalysis; α_level::Float64=0.05)
    println("\n" * "="^100)
    println("CONVERGENCE SLOPE ANALYSIS")
    println("="^100)
    println(@sprintf("%-8s %-12s %-18s %-18s %-20s", "Dim", "Method", "β (mean)", "β (±CI)", "Ratio βg/β0"))
    println("-"^100)
    
    for d in analysis.dimensions
        # TODO hacky strings
        fits_plain = analysis.results[d]["standard-warm"]
        fits_grad = analysis.results[d]["grads-warm"]
        
        β0_vals = [f.β for f in fits_plain]
        βg_vals = [f.β for f in fits_grad]
        
        β0_mean = mean(β0_vals)
        β0_std = std(β0_vals)
        
        βg_mean = mean(βg_vals)
        βg_std = std(βg_vals)
        
        ratio = βg_mean / β0_mean
        
        println(@sprintf("%-8d %-12s %+.6f (±%.6f) %+.6f (±%.6f)   %.6f", 
                         d, "standard-warm", β0_mean, β0_std, βg_mean, βg_std, ratio))
    end
    
    println("="^100)
    
    return nothing
end

"""
    theory_comparison_table(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)

Compare measured slope ratios to theory predictions.

Args:
    analysis: MultiRunAnalysis object
    r_c: Relative cost for theory prediction
    ν: Matérn parameter for theory prediction
"""
function theory_comparison_table(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)
    println("\n" * "="^120)
    println("THEORY COMPARISON: SLOPE RATIOS")
    println("="^120)
    println(@sprintf("%-8s %-20s %-20s %-20s %-20s", 
                     "Dim", "Measured βg/β0", "Hermite (theory)", "Cost advantage", "Unified pred."))
    println("-"^120)
    
    for d in analysis.dimensions
        # TODO hacky strings
        fits_plain = analysis.results[d]["standard-warm"]
        fits_grad = analysis.results[d]["grads-warm"]
        
        β0_mean = mean([f.β for f in fits_plain])
        βg_mean = mean([f.β for f in fits_grad])
        measured_ratio = βg_mean / β0_mean
        
        hermite_pred = theoretical_slope_ratio(d; ν=ν)
        cost_adv = theoretical_cost_advantage(d; r_c=r_c)
        unified_pred = theoretical_unified_slope_ratio(d; r_c=r_c, ν=ν)
        
        println(@sprintf("%-8d %-20.6f %-20.6f %-20.6f %-20.6f", 
                         d, measured_ratio, hermite_pred, cost_adv, unified_pred))
    end
    println("="^120)
    
    return nothing
end

# ============================================================================
# Data Loading
# ============================================================================

"""
    extract_problem_type_from_path(path::String)::String

Extract problem type from path like:
  - 'data/MultidimProblem{ABProblem}3/...'  → "ABProblem"
  - 'data/MeanGauss3/...'                    → "MeanGauss"
  
Returns the problem type (e.g., "ABProblem", "MeanGauss")
"""
function extract_problem_type_from_path(path::String)::String
    basename_path = basename(path)
    
    # Try pattern 1: MultidimProblem{ProblemType}d
    match_obj = match(r"MultidimProblem\{(\w+)\}", basename_path)
    if !isnothing(match_obj)
        return match_obj.captures[1]
    end
    
    # Try pattern 2: ProblemTyped (where ProblemType is letters, d is trailing digit(s))
    match_obj = match(r"^([A-Za-z]+)\d+$", basename_path)
    if !isnothing(match_obj)
        return match_obj.captures[1]
    end
    
    return "Unknown"
end

"""
    extract_dimension_from_path(path::String)::Int

Extract dimension from path like 'data/MultidimProblem{ABProblem}3/...'
Returns the numeric suffix (e.g., 3 for dimension 3)"""
function extract_dimension_from_path(path::String)::Int
    # Get the last component of the path (the directory/file name itself)
    basename_path = basename(path)
    # Extract number from end of directory name (e.g., "3" from "MultidimProblem{ABProblem}3")
    match_obj = match(r"(\d+)/?$", basename_path)
    if isnothing(match_obj)
        return 0
    end
    index = parse(Int, match_obj.captures[1])
    # Convert index to actual problem dimension (base problem is 2D, increments by 2)
    return 2 * index
end

"""
    load_experiment_data(data_dir::String, methods::Vector{String}=["standard-lazy", "grads-lazy"]; expected_runs::Int=20)::Dict

Load experiment data from "standard-lazy" and "grads-lazy" runs into a dictionary.

Args:
    data_dir: Root data directory containing subdirectories for each problem/dimension
    methods: List of method names to load (default: ["standard-lazy", "grads-lazy"])
    expected_runs: Expected number of runs for each method and dimension

Returns:
    Dict with structure: dim => {method => [scores_run1, scores_run2, ...]}
    where scores_i is a vector of scores for run i
"""
function load_experiment_data(data_dir::String, methods::Vector{String}=["standard-lazy", "grads-lazy"]; 
                              expected_runs::Int=20)::Dict
    data = Dict()
    
    # Check if data directory exists
    if !isdir(data_dir)
        error("Data directory does not exist: $data_dir")
    end
    
    # Find all subdirectories (one per problem/dimension)
    try
        problem_dirs = readdir(data_dir; join=true)
    catch e
        error("Failed to read data directory $data_dir: $e")
    end
    problem_dirs = filter(isdir, problem_dirs)
    problem_dirs = sort(problem_dirs)
    
    for problem_dir in problem_dirs
        dim = extract_dimension_from_path(problem_dir)
        
        for method in methods
            # Find all TVmetric files for this method
            method_prefix = "$(method)_"
            files = []  # Initialize before try block
            try
                all_files = readdir(problem_dir; join=true)
                files = sort([f for f in all_files if isfile(f) && contains(basename(f), method_prefix) && endswith(basename(f), "_TVmetric.jld2")])
            catch e
                @warn "Failed to read directory $problem_dir: $e"
                continue
            end
            
            if isempty(files)
                continue
            end
            
            # Extract run indices and sort by index
            run_indices = []
            for f in files
                fname = basename(f)
                parts = split(fname, "_")
                if length(parts) >= 2
                    run_idx = tryparse(Int, parts[2])
                    if !isnothing(run_idx)
                        push!(run_indices, run_idx)
                    end
                end
            end
            
            perm = sortperm(run_indices)
            files = files[perm]
            run_indices = run_indices[perm]
            
            # Load scores
            scores_list = Float64[]
            for f in files
                try
                    data_loaded = load(f)
                    if haskey(data_loaded, "score")
                        push!(scores_list, data_loaded["score"])
                    end
                catch e
                    @warn "Failed to load file $f: $e"
                end
            end
            
            if !isempty(scores_list)
                if !haskey(data, dim)
                    data[dim] = Dict()
                end
                data[dim][method] = [scores_list]  # Wrap in list for compatibility with analyze_multiple_runs
                
                if length(scores_list) != expected_runs
                    @warn "Dimension $dim, Method $method: loaded $(length(scores_list)) runs, expected $expected_runs"
                end
            end
        end
    end
    
    # Convert single score vectors to lists of runs for proper format
    # Each method should have a list of score vectors (one per run)
    for dim in keys(data)
        for method in keys(data[dim])
            scores_all = data[dim][method][1]  # This is a single vector of all scores
            # We need to convert this to the proper format
            # Since we loaded TVmetric files (one per run), scores_all already has all runs concatenated
            # We need to reshape this properly, but it seems the TVmetric file contains all iterations for one run
            # So scores_all is actually the convergence curve for one aggregated run
            # Let me reconsider the structure...
        end
    end
    
    return data
end

"""
    load_experiment_data_by_run(data_dir::String, methods::Vector{String}=["standard-lazy", "grads-lazy"]; expected_runs::Int=20)::Dict

Load experiment data aggregating runs properly.
Each run is stored in a separate TVmetric file.

Args:
    data_dir: Root data directory containing subdirectories for each problem/dimension
    methods: List of method names to load (default: ["standard-lazy", "grads-lazy"])
    expected_runs: Expected number of runs for each method and dimension

Returns:
    Dict with structure: dim => {method => [scores_run1, scores_run2, ...]}
    where scores_i is a vector of N scores for run i
"""
function load_experiment_data_by_run(data_dir::String, methods::Vector{String}=["standard-warm", "grads-warm"]; 
                                     expected_runs::Int=20)::Dict
    data = Dict()
    
    # Check if data directory exists
    if !isdir(data_dir)
        error("Data directory does not exist: $data_dir")
    end
    
    # Create method mapping if renaming
    method_map = Dict(m => m for m in methods)
    
    # Find all subdirectories (one per problem/dimension) using readdir instead of glob
    try
        problem_dirs = readdir(data_dir; join=true)
    catch e
        error("Failed to read data directory $data_dir: $e")
    end
    problem_dirs = filter(isdir, problem_dirs)
    problem_dirs = sort(problem_dirs)
    
    for problem_dir in problem_dirs
        dim = extract_dimension_from_path(problem_dir)
        if dim == 0
            continue  # Skip if dimension extraction failed
        end
        
        for method in methods
            method_name = method_map[method]
            scores_by_run = []
            
            # Find all TVmetric files for this method
            method_prefix = "$(method)_"
            files = []  # Initialize before try block
            try
                all_files = readdir(problem_dir; join=true)
                files = sort([f for f in all_files if isfile(f) && contains(basename(f), method_prefix) && endswith(basename(f), "_TVmetric.jld2")])
            catch e
                @warn "Failed to read directory $problem_dir: $e"
                continue
            end
            
            if isempty(files)
                continue
            end
            
            # Extract run indices and sort
            run_data = []
            for f in files
                fname = basename(f)
                parts = split(fname, "_")
                if length(parts) >= 2
                    run_idx = tryparse(Int, parts[2])
                    if !isnothing(run_idx)
                        push!(run_data, (run_idx, f))
                    end
                end
            end
            
            sort!(run_data; by=x->x[1])
            
            # Load scores for each run
            for (run_idx, file) in run_data
                try
                    data_loaded = load(file)
                    if haskey(data_loaded, "score")
                        score_vec = data_loaded["score"]
                        if score_vec isa Vector
                            push!(scores_by_run, score_vec)
                        else
                            push!(scores_by_run, [score_vec])
                        end
                    end
                catch e
                    @warn "Failed to load file $file: $e"
                end
            end
            
            if !isempty(scores_by_run)
                if !haskey(data, dim)
                    data[dim] = Dict()
                end
                data[dim][method_name] = scores_by_run
                
                if length(scores_by_run) != expected_runs
                    @warn "Dimension $dim, Method $method_name: loaded $(length(scores_by_run)) runs, expected $expected_runs"
                end
            end
        end
    end
    
    return data
end

"""
    load_experiment_data_by_run_filtered(data_dir::String, problem_type::String, methods::Vector{String}=["standard-warm", "grads-warm"]; expected_runs::Int=20)::Dict

Load experiment data for a specific problem type only.

Similar to load_experiment_data_by_run but filters to include only directories matching the given problem_type.

Args:
    data_dir: Root data directory
    problem_type: Problem type to filter for (e.g., "ABProblem", "SimpleProblem")
    methods: List of method names to load
    expected_runs: Expected number of runs per method

Returns:
    Dict with structure: dim => {method => [scores_run1, scores_run2, ...]}
    Only includes dimensions from the specified problem_type
"""
function load_experiment_data_by_run_filtered(data_dir::String, problem_type::String, methods::Vector{String}=["standard-warm", "grads-warm"]; 
                                              expected_runs::Int=20)::Dict
    data = Dict()
    
    # Check if data directory exists
    if !isdir(data_dir)
        error("Data directory does not exist: $data_dir")
    end
    
    # Create method mapping if renaming
    method_map = Dict(m => m for m in methods)
    
    # Find all subdirectories that match the problem_type
    try
        all_problem_dirs = readdir(data_dir; join=true)
    catch e
        error("Failed to read data directory $data_dir: $e")
    end
    all_problem_dirs = filter(isdir, all_problem_dirs)
    all_problem_dirs = sort(all_problem_dirs)
    
    # Filter to only directories matching this problem type
    problem_dirs = [d for d in all_problem_dirs if extract_problem_type_from_path(d) == problem_type]
    
    for problem_dir in problem_dirs
        dim = extract_dimension_from_path(problem_dir)
        if dim == 0
            continue  # Skip if dimension extraction failed
        end
        
        for method in methods
            method_name = method_map[method]
            scores_by_run = []
            
            # Find all TVmetric files for this method
            method_prefix = "$(method)_"
            files = []  # Initialize before try block
            try
                all_files = readdir(problem_dir; join=true)
                files = sort([f for f in all_files if isfile(f) && contains(basename(f), method_prefix) && endswith(basename(f), "_TVmetric.jld2")])
            catch e
                @warn "Failed to read directory $problem_dir: $e"
                continue
            end
            
            if isempty(files)
                continue
            end
            
            # Extract run indices and sort
            run_data = []
            for f in files
                fname = basename(f)
                parts = split(fname, "_")
                if length(parts) >= 2
                    run_idx = tryparse(Int, parts[2])
                    if !isnothing(run_idx)
                        push!(run_data, (run_idx, f))
                    end
                end
            end
            
            sort!(run_data; by=x->x[1])
            
            # Load scores for each run
            for (run_idx, file) in run_data
                try
                    data_loaded = load(file)
                    if haskey(data_loaded, "score")
                        score_vec = data_loaded["score"]
                        if score_vec isa Vector
                            push!(scores_by_run, score_vec)
                        else
                            push!(scores_by_run, [score_vec])
                        end
                    end
                catch e
                    @warn "Failed to load file $file: $e"
                end
            end
            
            if !isempty(scores_by_run)
                if !haskey(data, dim)
                    data[dim] = Dict()
                end
                data[dim][method_name] = scores_by_run
                
                if length(scores_by_run) != expected_runs
                    @warn "Dimension $dim, Method $method_name (Problem=$problem_type): loaded $(length(scores_by_run)) runs, expected $expected_runs"
                end
            end
        end
    end
    
    return data
end

# ============================================================================
# Main Analysis Function
# ============================================================================

"""
    main_analysis(data_dir::String; plot_output_dir::String="plots/convergence", r_c::Float64=2, ν::Float64=2.5, verbose::Bool=true)

Main function to orchestrate loading, analysis, and plotting of experiment data.

Args:
    data_dir: Root data directory containing experiment results
    plot_output_dir: Directory to save plots (default "plots/convergence")
    r_c: Relative cost for theory predictions (default 2. - holds for y_dim=1)
    ν: Matérn parameter for theory predictions (default 2.5)
    verbose: Print progress information

Returns:
    analysis::MultiRunAnalysis with all results
"""
function main_analysis(data_dir::String; 
                      plot_output_dir::String="plots/convergence",
                      r_c::Float64=2., 
                      ν::Float64=2.5,
                      verbose::Bool=true)
    
    # Load data
    if verbose
        println("\n" * "="^100)
        println("Loading experiment data...")
        println("="^100)
    end
    
    data = load_experiment_data_by_run(data_dir)
    
    if isempty(data)
        # Provide diagnostic information
        if !isdir(data_dir)
            error("Data directory does not exist: $data_dir")
        end
        
        try
            dirs = readdir(data_dir)
            if isempty(dirs)
                error("Data directory is empty: $data_dir")
            else
                sample_items = join(dirs[1:min(5,length(dirs))], ", ")
                extra = length(dirs) > 5 ? "..." : ""
                error("No data found in $data_dir\nFound $(length(dirs)) items: $sample_items$extra\nLooking for subdirectories with TVmetric files (standard-lazy_*_TVmetric.jld2 or grads-lazy_*_TVmetric.jld2)")
            end
        catch
            error("No data found in $data_dir")
        end
    end
    
    dims = sort(collect(keys(data)))
    if verbose
        println("Found dimensions: $dims")
        for d in dims
            methods = collect(keys(data[d]))
            println("  Dimension $d: $methods")
            for method in methods
                n_runs = length(data[d][method])
                println("    - $method: $n_runs runs")
            end
        end
    end
    
    # Analyze
    if verbose
        println("\n" * "="^100)
        println("Computing power law fits...")
        println("="^100)
    end
    
    analysis = analyze_multiple_runs(data; verbose=verbose, data_dir=data_dir)
    
    # Diagnostic output for problem types
    if verbose
        println("\nDiagnostic: Problem types detected: $(analysis.problem_types)")
        if isempty(analysis.problem_types)
            println("  → No problem types found. Listing directories in $data_dir:")
            try
                dirs = readdir(data_dir; join=true)
                dirs = filter(isdir, dirs)
                for d in dirs[1:min(5, length(dirs))]
                    prob_type = extract_problem_type_from_path(d)
                    println("    - $(basename(d)) → problem_type: '$prob_type'")
                end
                if length(dirs) > 5
                    println("    ... and $(length(dirs)-5) more directories")
                end
            catch
            end
        end
    end
    
    # Print summary
    if verbose
        summary_table(analysis)
    end
    
    # Print theory comparison
    if verbose
        theory_comparison_table(analysis; r_c=r_c, ν=ν)
    end
    
    # Create plots if CairoMakie is available
    if !isdir(plot_output_dir)
        mkpath(plot_output_dir)
    end
    
    if verbose
        println("\n" * "="^100)
        println("Generating plots...")
        println("="^100)
    end
    
    # Main slope comparison plot with critical iterations
    p_slopes = plot_slope_comparison(analysis; r_c=r_c, ν=ν)
    plot_file = joinpath(plot_output_dir, "slope_comparison.png")
    save(plot_file, p_slopes)
    if verbose
        println("Saved slope comparison plot to $plot_file")
    end
    
    # Detailed theory breakdown
    p_theory = plot_detailed_theory_analysis(analysis; r_c=r_c, ν=ν)
    plot_file = joinpath(plot_output_dir, "theory_breakdown.png")
    save(plot_file, p_theory)
    if verbose
        println("Saved detailed theory analysis plot to $plot_file")
    end
    
    # Cost-adjusted slope comparison
    p_cost_adjusted = plot_slope_comparison_cost_adjusted(analysis; r_c=r_c, ν=ν)
    plot_file = joinpath(plot_output_dir, "slope_comparison_cost_adjusted.png")
    save(plot_file, p_cost_adjusted)
    if verbose
        println("Saved cost-adjusted slope comparison plot to $plot_file")
    end
    
    # Slope comparison by problem type (if available)
    if !isempty(analysis.problem_types) && length(analysis.problem_types) > 0
        if verbose
            println("\nGenerating plots for each problem type independently...")
        end
        plot_all_main_figures_by_problem(analysis; output_dir=plot_output_dir, r_c=r_c, ν=ν, verbose=verbose)
    end
    
    # Plot individual convergence curves
    # for d in analysis.dimensions
    #     for method in ["standard-warm", "grads-warm"]
    #         if !haskey(analysis.results[d], method)
    #             continue
    #         end
    #         fits = analysis.results[d][method]
            
    #         if !isempty(fits)
    #             for (run_idx, fit) in enumerate(collect(fits)[1:min(3, length(fits))])  # Plot up to 3 representative runs
    #                 # Recover original scores from the fit object
    #                 original_scores = exp.(fit.log_scores)
    #                 p = plot_convergence_loglog(original_scores, label="Run $run_idx", 
    #                                                 title="Convergence curve: Dimension $d - $method")
    #                 plot_file = joinpath(plot_output_dir, "convergence_d$(d)_$(method)_run$(run_idx).png")
    #                 save(plot_file, p)
    #             end
    #             if verbose
    #                 println("Saved convergence plots for dimension $d - $method")
    #             end
    #         end
    #     end
    # end
    
    if verbose
        println("\n" * "="^100)
        println("Analysis complete!")
        println("="^100)
    end
    return analysis
end

# Plotting Helpers (optional, requires Plots.jl)
# ============================================================================

"""
    plot_slope_comparison_by_problem(analysis::MultiRunAnalysis; output_dir::String="plots", r_c::Float64=2.5, ν::Float64=2.5)

Plot measured slopes for each problem type separately.

Creates a 1x3 plot for each problem type showing:
- Plain BOSIP slopes vs dimension
- Grad-BOSIP slopes vs dimension
- Slope ratios with theoretical predictions

Each plot is saved as a separate PNG file: slope_comparison_{problem_type}.png

Args:
    analysis: MultiRunAnalysis object (must include problem type data)
    output_dir: Directory to save plots
    r_c: Relative cost for theory predictions
    ν: Matérn smoothness parameter

Returns:
    Dict mapping problem_type => Figure object
"""
function plot_slope_comparison_by_problem(analysis::MultiRunAnalysis; output_dir::String="plots", r_c::Float64=2.5, ν::Float64=2.5)
    problem_types = analysis.problem_types
    
    if isempty(problem_types)
        @warn "No problem type data available"
        return Dict()
    end
    
    figures = Dict()
    
    for problem_type in problem_types
        # Check if we have data for this problem type
        if !haskey(analysis.results_by_problem, problem_type) || isempty(analysis.results_by_problem[problem_type])
            continue
        end
        
        problem_results = analysis.results_by_problem[problem_type]
        dims = sort(collect(keys(problem_results))) .|> Int
        
        # Compute slopes for this problem type
        plain_s, plain_e, grad_s, grad_e, ratio_s, ratio_e = compute_slopes_from_results(problem_results, dims)
        
        # Compute theoretical predictions
        hermite_preds = [theoretical_slope_ratio(d; ν=ν) for d in dims]
        cost_preds = [theoretical_cost_advantage(d; r_c=r_c) for d in dims]
        unified_preds = [theoretical_unified_slope_ratio(d; r_c=r_c, ν=ν, regime=:pre_asymptotic) for d in dims]
        
        # Create figure with 1x3 grid
        fig = Figure(size=(1600, 400))
        
        # Left: Plain BOSIP slopes
        ax1 = Axis(fig[1, 1]; xlabel="d", ylabel="β", title="$problem_type: Plain BOSIP Slopes")
        errorbars!(ax1, dims, plain_s, plain_e; linewidth=2, color=:orange, whiskerwidth=3)
        scatter!(ax1, dims, plain_s; markersize=8, color=:orange)
        
        # Middle: Grad-BOSIP slopes
        ax2 = Axis(fig[1, 2]; xlabel="d", ylabel="β", title="$problem_type: Grad-BOSIP Slopes")
        errorbars!(ax2, dims, grad_s, grad_e; linewidth=2, color=:blue, whiskerwidth=3)
        scatter!(ax2, dims, grad_s; markersize=8, color=:blue)
        
        # Right: Slope ratios with theory
        ax3 = Axis(fig[1, 3]; xlabel="d", ylabel="βg/β0", title="$problem_type: Slope Ratio Comparison")
        errorbars!(ax3, dims, ratio_s, ratio_e; linewidth=2, color=:black, whiskerwidth=3)
        scatter!(ax3, dims, ratio_s; markersize=8, color=:black)
        lines!(ax3, dims, hermite_preds; label="Asymptotic", linestyle=:dash, linewidth=2, color=:blue)
        lines!(ax3, dims, cost_preds; label="Pre-asymptotic", linestyle=:dash, linewidth=2, color=:red)
        lines!(ax3, dims, unified_preds; label="Unified", linestyle=:solid, linewidth=2, color=:purple)
        axislegend(ax3; position=:lt)
        
        figures[problem_type] = fig
        
        # Save figure if output directory is provided
        if !isempty(output_dir)
            mkpath(output_dir)
            filename = joinpath(output_dir, "slope_comparison_$(problem_type).png")
            save(filename, fig)
            println("Saved: $filename")
        end
    end
    
    return figures
end

"""
    plot_all_main_figures_by_problem(analysis::MultiRunAnalysis; output_dir::String="plots", r_c::Float64=2.5, ν::Float64=2.5, verbose::Bool=false)

Generate all main comparative plots (slope comparison, theory analysis, cost-adjusted) for each problem type independently.

Creates subdirectory structure: output_dir/problem_type/{slope_comparison, theory_breakdown, cost_adjusted}_plots/

Args:
    analysis: MultiRunAnalysis object with results_by_problem data
    output_dir: Base output directory
    r_c: Relative cost for theory predictions
    ν: Matérn smoothness parameter
    verbose: Print progress information
"""
function plot_all_main_figures_by_problem(analysis::MultiRunAnalysis; output_dir::String="plots", 
                                          r_c::Float64=2.5, ν::Float64=2.5, verbose::Bool=false)
    problem_types = analysis.problem_types
    
    if isempty(problem_types)
        @warn "No problem type data available"
        return
    end
    
    for problem_type in problem_types
        # Check if we have data for this problem type
        if !haskey(analysis.results_by_problem, problem_type) || isempty(analysis.results_by_problem[problem_type])
            continue
        end
        
        problem_results = analysis.results_by_problem[problem_type]
        dims = sort(collect(keys(problem_results))) .|> Int
        
        # Create subdirectory for this problem type
        problem_dir = joinpath(output_dir, problem_type)
        mkpath(problem_dir)
        
        if verbose
            println("  Generating plots for problem type: $problem_type")
        end
        
        # Plot 1: Slope Comparison
        p_slopes = plot_slope_comparison_for_problem(problem_results, dims; r_c=r_c, ν=ν, problem_type=problem_type)
        plot_file = joinpath(problem_dir, "01_slope_comparison.png")
        save(plot_file, p_slopes)
        if verbose
            println("    Saved slope comparison: $plot_file")
        end
        
        # Plot 2: Detailed Theory Analysis
        p_theory = plot_detailed_theory_analysis_for_problem(problem_results, dims; r_c=r_c, ν=ν, problem_type=problem_type)
        plot_file = joinpath(problem_dir, "02_theory_breakdown.png")
        save(plot_file, p_theory)
        if verbose
            println("    Saved theory breakdown: $plot_file")
        end
        
        # Plot 3: Cost-Adjusted Slope Comparison
        p_cost_adjusted = plot_slope_comparison_cost_adjusted_for_problem(problem_results, dims; r_c=r_c, ν=ν, problem_type=problem_type)
        plot_file = joinpath(problem_dir, "03_slope_comparison_cost_adjusted.png")
        save(plot_file, p_cost_adjusted)
        if verbose
            println("    Saved cost-adjusted slopes: $plot_file")
        end
    end
    
    if verbose
        println("Completed generating independent problem-type plots")
    end
end

"""
    plot_slope_comparison_for_problem(problem_results::Dict, dims::Vector{Int}; r_c::Float64=2.5, ν::Float64=2.5, problem_type::String="")

Generate slope comparison plot for a single problem type.

Args:
    problem_results: Results dict for one problem type: {d => {method => [fit1, fit2, ...]}}
    dims: Vector of dimensions to plot
    r_c: Relative cost for theory predictions
    ν: Matérn smoothness parameter
    problem_type: Problem type name (for title)

Returns:
    Figure object
"""
function plot_slope_comparison_for_problem(problem_results::Dict, dims::Vector{Int}; 
                                          r_c::Float64=2.5, ν::Float64=2.5, problem_type::String="")
    # Compute theoretical predictions
    hermite_preds = [theoretical_slope_ratio(d; ν=ν) for d in dims]
    cost_preds = [theoretical_cost_advantage(d; r_c=r_c) for d in dims]
    unified_preds = [theoretical_unified_slope_ratio(d; r_c=r_c, ν=ν, regime=:pre_asymptotic) for d in dims]
    
    # Compute slopes for this problem
    plain_s, plain_e, grad_s, grad_e, ratio_s, ratio_e = compute_slopes_from_results(problem_results, dims)
    
    # Create figure with 1x3 grid
    fig = Figure(size=(1600, 400))
    
    title_prefix = isempty(problem_type) ? "" : "$problem_type: "
    
    # Left: Plain BOSIP slopes
    ax1 = Axis(fig[1, 1]; xlabel="d", ylabel="β", title="$(title_prefix)Plain BOSIP Slopes")
    errorbars!(ax1, dims, plain_s, plain_e; linewidth=2, color=:orange, whiskerwidth=3)
    scatter!(ax1, dims, plain_s; markersize=8, color=:orange)
    
    # Middle: Grad-BOSIP slopes
    ax2 = Axis(fig[1, 2]; xlabel="d", ylabel="β", title="$(title_prefix)Grad-BOSIP Slopes")
    errorbars!(ax2, dims, grad_s, grad_e; linewidth=2, color=:blue, whiskerwidth=3)
    scatter!(ax2, dims, grad_s; markersize=8, color=:blue)
    
    # Right: Slope ratios with theory
    ax3 = Axis(fig[1, 3]; xlabel="d", ylabel="βg/β0", title="$(title_prefix)Slope Ratio Comparison")
    errorbars!(ax3, dims, ratio_s, ratio_e; linewidth=2, color=:black, whiskerwidth=3)
    scatter!(ax3, dims, ratio_s; markersize=8, color=:black)
    lines!(ax3, dims, hermite_preds; label="Asymptotic", linestyle=:dash, linewidth=2, color=:blue)
    lines!(ax3, dims, cost_preds; label="Pre-asymptotic", linestyle=:dash, linewidth=2, color=:red)
    lines!(ax3, dims, unified_preds; label="Unified", linestyle=:solid, linewidth=2, color=:purple)
    axislegend(ax3; position=:lt)
    
    return fig
end

"""
    plot_detailed_theory_analysis_for_problem(problem_results::Dict, dims::Vector{Int}; r_c::Float64=2.5, ν::Float64=2.5, problem_type::String="")

Generate detailed theory analysis plot for a single problem type.

Args:
    problem_results: Results dict for one problem type
    dims: Vector of dimensions to plot
    r_c: Relative cost for theory predictions
    ν: Matérn smoothness parameter
    problem_type: Problem type name (for title)

Returns:
    Figure object
"""
function plot_detailed_theory_analysis_for_problem(problem_results::Dict, dims::Vector{Int}; 
                                                   r_c::Float64=2.5, ν::Float64=2.5, problem_type::String="")
    # Measured ratios
    measured_ratios = Float64[]
    for d in dims
        if haskey(problem_results[d], "standard-warm") && haskey(problem_results[d], "grads-warm")
            β0_mean = mean([f.β for f in problem_results[d]["standard-warm"]])
            βg_mean = mean([f.β for f in problem_results[d]["grads-warm"]])
            push!(measured_ratios, βg_mean / β0_mean)
        end
    end
    
    # Theory components
    hermite_factors = [theoretical_slope_ratio(d; ν=ν) for d in dims]
    cost_factors = [theoretical_cost_advantage(d; r_c=r_c) for d in dims]
    unified_preds = [theoretical_unified_slope_ratio(d; r_c=r_c, ν=ν, regime=:pre_asymptotic) for d in dims]
    
    fig = Figure(size=(1200, 500))
    
    title_prefix = isempty(problem_type) ? "" : "$problem_type: "
    
    # Determine if we can use log scale (all measured ratios must be positive and reasonable)
    use_log_scale = !isempty(measured_ratios) && all(measured_ratios .> 0.01)
    
    # Left: Individual factors
    ax1_kwargs = Dict{Symbol, Any}(
        :xlabel => "Dimension (d)",
        :ylabel => "Factor magnitude",
        :title => "$(title_prefix)Theory Components"
    )
    if use_log_scale
        ax1_kwargs[:yscale] = log10
    end
    ax1 = Axis(fig[1, 1]; ax1_kwargs...)
    
    lines!(ax1, dims, hermite_factors; label="Hermite factor (≈1 + 2d/[(2ν+d)(2ν+2d+2)])", 
           linestyle=:dash, linewidth=2, color=:blue)
    lines!(ax1, dims, cost_factors; label="Cost factor ((1+d)/r_c)", 
           linestyle=:dash, linewidth=2, color=:red)
    lines!(ax1, dims, unified_preds; label="Product (unified)", 
           linestyle=:solid, linewidth=2, color=:purple)
    # Only scatter measured ratios if they're all positive
    if !isempty(measured_ratios) && all(measured_ratios .> 0)
        scatter!(ax1, dims, measured_ratios; label="Measured", markersize=8, color=:black)
    elseif !isempty(measured_ratios)
        # Plot as text warning if not all positive
        text!(ax1, 0.5, 0.95, text="⚠ Some measured ratios negative", space=:relative, fontsize=10, color=:red)
    end
    
    axislegend(ax1; position=:lt)
    
    # Right: Breakdown composition
    ax2 = Axis(fig[1, 2];
               xlabel="Dimension (d)",
               ylabel="Slope Ratio Contribution",
               title="$(title_prefix)Asymptotic vs Pre-asymptotic Dominance")
    
    # Show how much of unity (asymptotic) vs (1+d)/r_c (pre-asymptotic) is visible
    hermite_contribution = hermite_factors .- 1  # How much above the baseline
    cost_contribution = cost_factors .- 1        # How much above baseline from cost
    
    barplot!(ax2, dims, hermite_contribution; label="Hermite contribution", 
             color=:blue, alpha=0.6, dodge=1, width=0.4)
    barplot!(ax2, dims .- 0.2, cost_contribution; label="Cost advantage contribution",
             color=:red, alpha=0.6, dodge=1, width=0.4)
    
    axislegend(ax2; position=:lt)
    
    return fig
end

"""
    plot_slope_comparison_cost_adjusted_for_problem(problem_results::Dict, dims::Vector{Int}; r_c::Float64=2.5, ν::Float64=2.5, problem_type::String="")

Generate cost-adjusted slope comparison plot for a single problem type.

Args:
    problem_results: Results dict for one problem type
    dims: Vector of dimensions to plot
    r_c: Relative cost for theory predictions
    ν: Matérn smoothness parameter
    problem_type: Problem type name (for title)

Returns:
    Figure object
"""
function plot_slope_comparison_cost_adjusted_for_problem(problem_results::Dict, dims::Vector{Int}; 
                                                        r_c::Float64=2.5, ν::Float64=2.5, problem_type::String="")
    # Cost-adjusted measured slope ratios
    measured_ratios_adjusted = Float64[]
    measured_errors_adjusted = Float64[]
    measured_ratios_plain = Float64[]
    measured_errors_plain = Float64[]
    
    for d in dims
        if haskey(problem_results[d], "standard-warm") && haskey(problem_results[d], "grads-warm")
            # Per-simulation slopes (original)
            β0_vals_plain = [f.β for f in problem_results[d]["standard-warm"]]
            βg_vals_plain = [f.β for f in problem_results[d]["grads-warm"]]
            
            # Recover original scores and refit on cost-adjusted basis
            β0_vals_adjusted = Float64[]
            βg_vals_adjusted = Float64[]
            
            for fit_plain in problem_results[d]["standard-warm"]
                scores_plain = exp.(fit_plain.log_scores)
                fit_adjusted = fit_powerlaw_cost_adjusted(scores_plain, d, r_c; for_grad=false, verbose=false)
                push!(β0_vals_adjusted, fit_adjusted.β)
            end
            
            for fit_grad in problem_results[d]["grads-warm"]
                scores_grad = exp.(fit_grad.log_scores)
                fit_adjusted = fit_powerlaw_cost_adjusted(scores_grad, d, r_c; for_grad=true, verbose=false)
                push!(βg_vals_adjusted, fit_adjusted.β)
            end
            
            # Compute plain ratios
            β0_mean_plain = mean(β0_vals_plain)
            βg_mean_plain = mean(βg_vals_plain)
            ratio_plain = βg_mean_plain / β0_mean_plain
            push!(measured_ratios_plain, ratio_plain)
            individual_ratios_plain = βg_vals_plain ./ β0_vals_plain
            push!(measured_errors_plain, std(individual_ratios_plain))
            
            # Compute cost-adjusted ratios
            β0_mean_adjusted = mean(β0_vals_adjusted)
            βg_mean_adjusted = mean(βg_vals_adjusted)
            ratio_adjusted = βg_mean_adjusted / β0_mean_adjusted
            push!(measured_ratios_adjusted, ratio_adjusted)
            individual_ratios_adjusted = βg_vals_adjusted ./ β0_vals_adjusted
            push!(measured_errors_adjusted, std(individual_ratios_adjusted))
        end
    end
    
    # Theoretical predictions
    hermite_preds = [theoretical_slope_ratio(d; ν=ν) for d in dims]
    
    fig = Figure(size=(1400, 550))
    
    title_prefix = isempty(problem_type) ? "" : "$problem_type: "
    
    # Left panel: cost-adjusted slopes
    ax1 = Axis(fig[1, 1]; 
               xlabel="Dimension (d)", 
               ylabel="Slope Ratio: βg / β₀", 
               title="$(title_prefix)Cost-Adjusted Slope Ratios\n(log(error) vs log(n × (1+d)) for gradient, log(n) for plain)")
    
    errorbars!(ax1, dims, measured_ratios_adjusted, measured_errors_adjusted; 
               label="Cost-adjusted βg/β₀", 
               linewidth=2.5, color=:black, whiskerwidth=3)
    scatter!(ax1, dims, measured_ratios_adjusted; markersize=10, color=:black)
    
    # Overlay Hermite prediction (asymptotic theory)
    lines!(ax1, dims, hermite_preds; label="Hermite theory (asymptotic)", 
           linestyle=:dash, linewidth=2, color=:blue)
    
    # Reference line at 1
    hlines!(ax1, 1.0; label="Equal slopes", 
           linestyle=:dot, linewidth=1.5, color=:gray)
    
    axislegend(ax1; position=:lt)
    
    # Right panel: comparison of plain vs cost-adjusted
    ax2 = Axis(fig[1, 2];
               xlabel="Dimension (d)",
               ylabel="Slope Ratio",
               title="$(title_prefix)Per-Simulation vs Cost-Adjusted Slopes")
    
    errorbars!(ax2, dims .- 0.15, measured_ratios_plain, measured_errors_plain; 
               label="Per-simulation (original)", 
               linewidth=2, color=:orange, whiskerwidth=2)
    scatter!(ax2, dims .- 0.15, measured_ratios_plain; markersize=8, color=:orange)
    
    errorbars!(ax2, dims .+ 0.15, measured_ratios_adjusted, measured_errors_adjusted; 
               label="Cost-adjusted", 
               linewidth=2, color=:blue, whiskerwidth=2)
    scatter!(ax2, dims .+ 0.15, measured_ratios_adjusted; markersize=8, color=:blue)
    
    # Overlay Hermite theory prediction
    lines!(ax2, dims, hermite_preds; label="Hermite theory", 
           linestyle=:dash, linewidth=2.5, color=:purple)
    
    hlines!(ax2, 1.0; linestyle=:dot, linewidth=1.5, color=:gray)
    
    axislegend(ax2; position=:lt)
    
    return fig
end

"""
    plot_convergence_loglog(scores::Vector; label::String="", title::String="")

Plot convergence curve on log-log scale with power law fit overlay.

Args:
    scores: Score vector
    label: Label for the line
    title: Plot title

Returns:
    Figure object (uses CairoMakie)
"""
function plot_convergence_loglog(scores::Vector; label::String="", title::String="")
    fit = fit_powerlaw(scores)
    n = length(scores)
    
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale=log10, yscale=log10, 
              xlabel="Iteration", ylabel="Score", title=title)
    
    # Plot observed scores
    lines!(ax, 1:n, scores; label=label, linewidth=2)
    
    # Overlay fitted power law
    n_plot = 10 .^ LinRange(0, log10(n), 100)
    scores_fit = exp.(fit.α .+ fit.β .* log.(n_plot))
    lines!(ax, n_plot, scores_fit; label="Power law fit (β=$(round(fit.β; digits=3)))", 
           linestyle=:dash, linewidth=2)
    
    axislegend(ax; position=:lt)
    return fig
end

"""
    fit_powerlaw_cost_adjusted(scores::Vector, d::Int, r_c::Float64; for_grad::Bool=true, verbose::Bool=false)

Fit power law to convergence curve, adjusted for computational cost.

For gradient method: transforms x = n (simulation count) to x_cost = n * (1+d)
                     (effective observations, which scales with the actual information gained)

For plain method: uses x = n directly (1 observation per simulation)

Args:
    scores: Vector of decreasing error values
    d: Problem dimension (used for gradient adjustment factor)
    r_c: Relative cost (gradient cost / plain cost)
    for_grad: If true, adjust for gradient cost; if false, fit plain method
    verbose: Print fit details

Returns:
    PowerLawFit object, but with log_n representing the adjusted iteration count
"""
function fit_powerlaw_cost_adjusted(scores::Vector, d::Int, r_c::Float64; for_grad::Bool=true, verbose::Bool=false)
    n = length(scores)
    
    if any(scores .<= 0)
        error("All scores must be positive for log-log fit")
    end
    
    log_scores = log.(scores)
    
    # Adjust iteration count for effective observations (gradient has 1+d observations per call)
    if for_grad
        # Gradient: n simulations → n*(1+d) effective observations
        adjusted_n = (1:n) .* (1 + d)
    else
        # Plain: n simulations → n observations
        adjusted_n = 1:n
    end
    
    log_n_adjusted = log.(adjusted_n)
    
    # Fit linear regression on cost-adjusted basis
    X = hcat(ones(n), log_n_adjusted)
    β_est = X \ log_scores
    
    α, β = β_est[1], β_est[2]
    
    fitted = α .+ β .* log_n_adjusted
    residuals = log_scores .- fitted
    
    σ_squared = sum(residuals .^ 2) / (n - 2)
    σ = sqrt(σ_squared)
    
    df = (; log_n = log_n_adjusted, log_scores = log_scores)
    model = lm(@formula(log_scores ~ log_n), df)
    
    if verbose
        method = for_grad ? "Gradient (cost-adjusted)" : "Plain (standard)"
        println("Power Law Fit ($method)")
        println("=" ^ 50)
        println("Model: log(score) = α + β * log(n_eff)")
        println("  n_eff = n × $(for_grad ? "(1+d)" : "1") for $(for_grad ? "gradient" : "plain")")
        println("Fitted values:")
        println("  α (intercept) = $(round(α; digits=6))")
        println("  β (slope)     = $(round(β; digits=6))")
        println("  σ (std err)   = $(round(σ; digits=6))")
        println("=" ^ 50)
    end
    
    return PowerLawFit(α, β, n, log_n_adjusted, log_scores, residuals, σ, n-2, model)
end

"""
    critical_iteration_count(d::Int; ν::Float64=2.5)

Estimate the critical sample size N_crit ≈ d^(d/ν) where GP transitions from 
underfitting (flat convergence) to learning phase.

Args:
    d: Dimension
    ν: Matérn smoothness parameter

Returns:
    Estimated critical sample size
"""
function critical_iteration_count(d::Int; ν::Float64=2.5)
    return Int(round(d ^ (d / ν)))
end

"""
    plot_slope_comparison(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)

Plot measured slopes for BOSIP and grad-BOSIP vs dimension, along with slope ratios.

**IMPORTANT: Shows slopes per SIMULATION COUNT, not cost-adjusted.**
See `plot_slope_comparison_cost_adjusted()` for computation-cost-normalized comparison.

Creates a comprehensive visualization showing:
- Plain BOSIP slopes vs dimension
- Grad-BOSIP slopes vs dimension
- Measured slope ratios with theoretical predictions (Hermite, pre-asymptotic, unified)

Args:
    analysis: MultiRunAnalysis object
    r_c: Relative cost of gradient evaluation (default 2.5 for AD)
    ν: Matérn smoothness parameter (default 2.5 for Matérn-5/2)

Returns:
    Figure object (uses CairoMakie)
"""
function compute_slopes_from_results(results::Dict, dims::Vector{Int})
    """Helper function to compute slopes and errors from a results dictionary."""
    plain_slopes = Float64[]
    plain_errors = Float64[]
    grad_slopes = Float64[]
    grad_errors = Float64[]
    measured_ratios = Float64[]
    measured_errors = Float64[]
    
    for d in dims
        if haskey(results[d], "standard-warm") && haskey(results[d], "grads-warm")
            fits_plain = results[d]["standard-warm"]
            fits_grad = results[d]["grads-warm"]

            β0_vals = [f.β for f in fits_plain]
            βg_vals = [f.β for f in fits_grad]
            
            β0_mean = mean(β0_vals)
            βg_mean = mean(βg_vals)
            
            push!(plain_slopes, β0_mean)
            push!(plain_errors, std(β0_vals))
            push!(grad_slopes, βg_mean)
            push!(grad_errors, std(βg_vals))
            
            # Mean slope ratio
            ratio_mean = βg_mean / β0_mean
            push!(measured_ratios, ratio_mean)
            
            # Uncertainty estimate: standard deviation of individual run ratios
            individual_ratios = βg_vals ./ β0_vals
            ratio_std = std(individual_ratios)
            push!(measured_errors, ratio_std)
        end
    end
    
    return plain_slopes, plain_errors, grad_slopes, grad_errors, measured_ratios, measured_errors
end

function plot_slope_comparison(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)
    dims = analysis.dimensions
    
    # Compute theoretical predictions
    hermite_preds = Float64[]
    cost_preds = Float64[]
    unified_preds = Float64[]
    
    for d in dims
        push!(hermite_preds, theoretical_slope_ratio(d; ν=ν))
        push!(cost_preds, theoretical_cost_advantage(d; r_c=r_c))
        push!(unified_preds, theoretical_unified_slope_ratio(d; r_c=r_c, ν=ν, regime=:pre_asymptotic))
    end
    
    # Compute slopes from all problems combined
    plain_s, plain_e, grad_s, grad_e, ratio_s, ratio_e = compute_slopes_from_results(analysis.results, dims)
    
    # Create figure with 1x3 grid
    fig = Figure(size=(1600, 400))
    
    # Left: Plain BOSIP slopes
    ax1 = Axis(fig[1, 1]; xlabel="d", ylabel="β", title="Plain BOSIP Slopes")
    errorbars!(ax1, dims, plain_s, plain_e; linewidth=2, color=:orange, whiskerwidth=3)
    scatter!(ax1, dims, plain_s; markersize=8, color=:orange)
    
    # Middle: Grad-BOSIP slopes
    ax2 = Axis(fig[1, 2]; xlabel="d", ylabel="β", title="Grad-BOSIP Slopes")
    errorbars!(ax2, dims, grad_s, grad_e; linewidth=2, color=:blue, whiskerwidth=3)
    scatter!(ax2, dims, grad_s; markersize=8, color=:blue)
    
    # Right: Slope ratios with theory
    ax3 = Axis(fig[1, 3]; xlabel="d", ylabel="βg/β0", title="Slope Ratio Comparison")
    errorbars!(ax3, dims, ratio_s, ratio_e; linewidth=2, color=:black, whiskerwidth=3)
    scatter!(ax3, dims, ratio_s; markersize=8, color=:black)
    lines!(ax3, dims, hermite_preds; label="Asymptotic", linestyle=:dash, linewidth=2, color=:blue)
    lines!(ax3, dims, cost_preds; label="Pre-asymptotic", linestyle=:dash, linewidth=2, color=:red)
    lines!(ax3, dims, unified_preds; label="Unified", linestyle=:solid, linewidth=2, color=:purple)
    axislegend(ax3; position=:lt)
    
    return fig
end

"""
    plot_detailed_theory_analysis(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)

Create detailed analysis plot showing the breakdown of theoretical contributions.

Plots:
- Measured slope ratios with uncertainty
- Hermite factor (asymptotic order correction)
- Cost factor (1+d)/r_c (pre-asymptotic phase transition)
- Their product (unified prediction)

Args:
    analysis: MultiRunAnalysis object
    r_c: Relative cost
    ν: Matérn parameter

Returns:
    Figure object
"""
function plot_detailed_theory_analysis(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)
    dims = analysis.dimensions
    
    # Measured ratios
    measured_ratios = Float64[]
    for d in dims
        β0_mean = mean([f.β for f in analysis.results[d]["standard-warm"]])
        βg_mean = mean([f.β for f in analysis.results[d]["grads-warm"]])
        push!(measured_ratios, βg_mean / β0_mean)
    end
    
    # Theory components
    hermite_factors = [theoretical_slope_ratio(d; ν=ν) for d in dims]
    cost_factors = [theoretical_cost_advantage(d; r_c=r_c) for d in dims]
    unified_preds = [theoretical_unified_slope_ratio(d; r_c=r_c, ν=ν, regime=:pre_asymptotic) for d in dims]
    
    fig = Figure(size=(1200, 500))
    
    # Left: Individual factors
    ax1 = Axis(fig[1, 1];
               xlabel="Dimension (d)",
               ylabel="Factor magnitude",
               title="Theory Components",
               yscale=log10)
    
    lines!(ax1, dims, hermite_factors; label="Hermite factor (≈1 + 2d/[(2ν+d)(2ν+2d+2)])", 
           linestyle=:dash, linewidth=2, color=:blue)
    lines!(ax1, dims, cost_factors; label="Cost factor ((1+d)/r_c)", 
           linestyle=:dash, linewidth=2, color=:red)
    lines!(ax1, dims, unified_preds; label="Product (unified)", 
           linestyle=:solid, linewidth=2, color=:purple)
    scatter!(ax1, dims, measured_ratios; label="Measured", markersize=8, color=:black)
    
    axislegend(ax1; position=:lt)
    
    # Right: Breakdown composition
    ax2 = Axis(fig[1, 2];
               xlabel="Dimension (d)",
               ylabel="Slope Ratio Contribution",
               title="Asymptotic vs Pre-asymptotic Dominance")
    
    # Show how much of unity (asymptotic) vs (1+d)/r_c (pre-asymptotic) is visible
    hermite_contribution = hermite_factors .- 1  # How much above the baseline
    cost_contribution = cost_factors .- 1        # How much above baseline from cost
    
    barplot!(ax2, dims, hermite_contribution; label="Hermite contribution", 
             color=:blue, alpha=0.6, dodge=1, width=0.4)
    barplot!(ax2, dims .- 0.2, cost_contribution; label="Cost advantage contribution",
             color=:red, alpha=0.6, dodge=1, width=0.4)
    
    axislegend(ax2; position=:lt)
    
    return fig
end

"""
    plot_slope_comparison_cost_adjusted(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)

Plot measured slope ratios ADJUSTED for effective observations per method.

**Cost-adjusted basis:**
- Plain method: log(error) = α₀ + β₀ × log(n) where n = simulation count
- Gradient method: log(error) = α_g + β_g × log(n × (1+d)) where n = simulation count, (1+d) = effective observations per simulation

**Key insight:** When cost-adjusting, we measure slopes on an equal footing:
both methods' slopes now reflect convergence rate per *observation received*.
The gradient method's higher slope (when cost-adjusted) reflects its ability
to achieve better function approximation with each observation.

Args:
    analysis: MultiRunAnalysis object
    r_c: Relative cost
    ν: Matérn parameter

Returns:
    Figure object comparing cost-adjusted measured slopes vs theory
"""
function plot_slope_comparison_cost_adjusted(analysis::MultiRunAnalysis; r_c::Float64=2.5, ν::Float64=2.5)
    dims = analysis.dimensions
    
    # Cost-adjusted measured slope ratios
    measured_ratios_adjusted = Float64[]
    measured_errors_adjusted = Float64[]
    
    # Per-simulation measured slope ratios (for comparison)
    measured_ratios_plain = Float64[]
    measured_errors_plain = Float64[]
    
    for d in dims
        # Per-simulation slopes (original, not cost-adjusted)
        β0_vals_plain = [f.β for f in analysis.results[d]["standard-warm"]]
        βg_vals_plain = [f.β for f in analysis.results[d]["grads-warm"]]
        
        # Recover original scores and refit on cost-adjusted basis
        β0_vals_adjusted = Float64[]
        βg_vals_adjusted = Float64[]
        
        for fit_plain in analysis.results[d]["standard-warm"]
            scores_plain = exp.(fit_plain.log_scores)
            fit_adjusted = fit_powerlaw_cost_adjusted(scores_plain, d, r_c; for_grad=false, verbose=false)
            push!(β0_vals_adjusted, fit_adjusted.β)
        end
        
        for fit_grad in analysis.results[d]["grads-warm"]
            scores_grad = exp.(fit_grad.log_scores)
            fit_adjusted = fit_powerlaw_cost_adjusted(scores_grad, d, r_c; for_grad=true, verbose=false)
            push!(βg_vals_adjusted, fit_adjusted.β)
        end
        
        # Compute plain (per-simulation) ratios
        β0_mean_plain = mean(β0_vals_plain)
        βg_mean_plain = mean(βg_vals_plain)
        ratio_plain = βg_mean_plain / β0_mean_plain
        push!(measured_ratios_plain, ratio_plain)
        individual_ratios_plain = βg_vals_plain ./ β0_vals_plain
        push!(measured_errors_plain, std(individual_ratios_plain))
        
        # Compute cost-adjusted ratios
        β0_mean_adjusted = mean(β0_vals_adjusted)
        βg_mean_adjusted = mean(βg_vals_adjusted)
        ratio_adjusted = βg_mean_adjusted / β0_mean_adjusted
        push!(measured_ratios_adjusted, ratio_adjusted)
        individual_ratios_adjusted = βg_vals_adjusted ./ β0_vals_adjusted
        push!(measured_errors_adjusted, std(individual_ratios_adjusted))
    end
    
    # Theoretical predictions
    hermite_preds = [theoretical_slope_ratio(d; ν=ν) for d in dims]
    
    fig = Figure(size=(1400, 550))
    
    # Left panel: cost-adjusted slopes
    ax1 = Axis(fig[1, 1]; 
               xlabel="Dimension (d)", 
               ylabel="Slope Ratio: βg / β₀", 
               title="Cost-Adjusted Slope Ratios\n(log(error) vs log(n × (1+d)) for gradient, log(n) for plain)")
    
    errorbars!(ax1, dims, measured_ratios_adjusted, measured_errors_adjusted; 
               label="Cost-adjusted βg/β₀", 
               linewidth=2.5, color=:black, whiskerwidth=3)
    scatter!(ax1, dims, measured_ratios_adjusted; markersize=10, color=:black)
    
    # Overlay Hermite prediction (asymptotic theory)
    lines!(ax1, dims, hermite_preds; label="Hermite theory (asymptotic)", 
           linestyle=:dash, linewidth=2, color=:blue)
    
    # Reference line at 1
    hlines!(ax1, 1.0; label="Equal slopes", 
           linestyle=:dot, linewidth=1.5, color=:gray)
    
    axislegend(ax1; position=:lt)
    
    # Right panel: comparison of plain vs cost-adjusted
    ax2 = Axis(fig[1, 2];
               xlabel="Dimension (d)",
               ylabel="Slope Ratio",
               title="Per-Simulation vs Cost-Adjusted Slopes")
    
    errorbars!(ax2, dims .- 0.15, measured_ratios_plain, measured_errors_plain; 
               label="Per-simulation (original)", 
               linewidth=2, color=:orange, whiskerwidth=2)
    scatter!(ax2, dims .- 0.15, measured_ratios_plain; markersize=8, color=:orange)
    
    errorbars!(ax2, dims .+ 0.15, measured_ratios_adjusted, measured_errors_adjusted; 
               label="Cost-adjusted", 
               linewidth=2, color=:blue, whiskerwidth=2)
    scatter!(ax2, dims .+ 0.15, measured_ratios_adjusted; markersize=8, color=:blue)
    
    # Overlay Hermite theory prediction
    lines!(ax2, dims, hermite_preds; label="Hermite theory", 
           linestyle=:dash, linewidth=2.5, color=:purple)
    
    hlines!(ax2, 1.0; linestyle=:dot, linewidth=1.5, color=:gray)
    
    axislegend(ax2; position=:lt)
    
    return fig
end

# ============================================================================
# Usage Examples
# ============================================================================
"""
    Example usage of the asymptotics_analytics module:

    # Load and analyze experiment data
    data_dir = "data"  # root directory containing problem subdirectories
    analysis = main_analysis(data_dir; 
                            plot_output_dir="plots",
                            r_c=2.5,
                            ν=2.5,
                            verbose=true)

    # The returned analysis object can be used for further analysis:
    # - access.dimensions for list of dimensions
    # - analysis.results[d][method] for PowerLawFit objects for dimension d and method name
    # - Each PowerLawFit contains .α, .β, .σ and other fit parameters

    # Manual data loading (if you need more control):
    data = load_experiment_data_by_run("data"; 
                                       methods=["standard-warm", "grads-warm"],
                                       expected_runs=20)
    
    # Analyze the data
    analysis = analyze_multiple_runs(data; verbose=true)
    
    # Print results
    summary_table(analysis)
    theory_comparison_table(analysis; r_c=2.5, ν=2.5)
    
    # Get confidence intervals for a particular fit
    fit = analysis.results[2]["standard-warm"][1]  # First fit for dimension 2, plain method
    ci = confidence_interval_slope(fit; α_level=0.05)
    println("Slope β = \$(fit.β) with 95% CI: \$ci")
"""
