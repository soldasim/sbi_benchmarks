
using BOSIP

function l2_norm(f_approx, f_true_vals::AbstractMatrix{<:Real}, xs::AbstractMatrix{<:Real}, log_ws::AbstractVector{<:Real})
    # Evaluate approximate simulator at all sample points (xs columns are samples)
    f_approx_vals = f_approx(xs)
    
    # Compute weighted L2 norm using importance weights in log space for numerical stability
    squared_diffs = (f_true_vals .- f_approx_vals) .^ 2
    
    # Compute weighted L2 norm for each output dimension using logmeanexp
    # result[d] = sqrt(mean_w(squared_diffs[d,i])) where mean_w is weighted mean
    n_dims = size(squared_diffs, 1)
    l2_norm_val = similar(squared_diffs, n_dims)
    
    for d = 1:n_dims
        # Compute weighted mean in log space: log(sum(w_i * x_i) / sum(w_i))
        # = logsumexp(log_w_i + log_x_i) - logsumexp(log_w_i)
        log_weighted_vals = log_ws .+ log.(squared_diffs[d, :])
        log_weighted_sum = BOSIP.logsumexp(log_weighted_vals)
        log_weight_sum = BOSIP.logsumexp(log_ws)
        log_normalized_mean = log_weighted_sum - log_weight_sum
        l2_norm_val[d] = sqrt(exp(log_normalized_mean))
    end
    
    return vec(l2_norm_val) # always a vector of length y_dim
end

"""
    ConvergenceCallback{M<:Function}

Callback that tracks convergence of the learned simulator function using a predefined convergence metric.

Similar to MetricCallback in BOSIP.jl, but instead of measuring posterior divergence, it measures
how well the current approximation of the simulator matches the true simulator on a predefined grid.

Fields:
- `simulator_approx_fn::Function`: Function that returns an approximation of the simulator given the current optimizer state
- `convergence_metric::Function`: Function(f_approx, true_sim_outputs, xs, log_ws) that computes convergence metric (e.g., l2_norm)
- `xs::Matrix{Float64}`: Grid points (columns are samples) sampled from the prior
- `log_ws::Vector{Float64}`: Log-weights (negative log prior densities) for importance weighting
- `true_sim_outputs::Matrix{Float64}`: True simulator outputs at the grid points
- `score_history::Matrix{Float64}`: History of convergence scores across iterations. Rows = output dimensions, Columns = iterations
"""
@kwdef mutable struct ConvergenceCallback <: BosipCallback
    simulator_approx_fn::Function = sim_approx
    convergence_metric::Function = l2_norm
    xs::Matrix{Float64}
    log_ws::Vector{Float64}
    true_sim_outputs::Matrix{Float64}
    score_history::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)  # (n_dims, n_iterations)
end

function sim_approx(bosip::BosipProblem)
    m = model_posterior(bosip.problem)
    return x -> mean(m, x)
end

function (cb::ConvergenceCallback)(problem::BosipProblem; first::Bool, options::BossOptions, kwargs...)
    if first && !isempty(cb.score_history)
        options.info && @warn "A continued run detected. Not calculating the first convergence score to avoid duplicates."
        return
    end

    # Get the current approximate simulator
    approx_sim = cb.simulator_approx_fn(problem)
    
    # Compute convergence score
    score = cb.convergence_metric(approx_sim, cb.true_sim_outputs, cb.xs, cb.log_ws)
    options.info && @show score
    
    # Append score as a new column to score_history
    if isempty(cb.score_history)
        # First iteration: initialize the matrix
        if score isa AbstractVector
            cb.score_history = reshape(score, length(score), 1)
        else
            cb.score_history = reshape([score], 1, 1)
        end
    else
        # Subsequent iterations: append as a new column
        if score isa AbstractVector
            cb.score_history = hcat(cb.score_history, score)
        else
            cb.score_history = hcat(cb.score_history, [score])
        end
    end
end

"""
    ConvergenceCallback(problem::AbstractProblem; 
                       simulator_approx_fn::Function,
                       convergence_metric::Function = l2_norm)

Create a ConvergenceCallback for tracking simulator convergence during optimization.

Arguments:
- `problem::AbstractProblem`: The inference problem
- `simulator_approx_fn`: Function or method to extract approximate simulator from BosipProblem
- `convergence_metric`: Convergence metric function (default: l2_norm)

Example:
```julia
cb = ConvergenceCallback(
    problem,
    simulator_approx_fn = prob -> prob.model,
    convergence_metric = l2_norm
)
```
"""
function ConvergenceCallback(problem::AbstractProblem; 
                            simulator_approx_fn::Function,
                            convergence_metric::Function = l2_norm)
    # Load precomputed simulator grid
    grid = load_simulator_grid(problem)
    xs = grid.xs
    log_ws = grid.log_ws
    true_sim_outputs = grid.sim_outputs
    
    return ConvergenceCallback(;
        simulator_approx_fn,
        convergence_metric,
        xs,
        log_ws,
        true_sim_outputs,
    )
end
