include("plot_slopes_uniform.jl")

println("\n=== AB problem gap debug (regression-based) ===")
for s in 1:3
    p = MultidimProblem(ABProblem(), s)
    d = 2 * s
    for metric in [:tv, :convergence]
        xs_std,   ys_std   = median_scores_for_group(p, "uniform";       metric=metric)
        xs_grads, ys_grads = median_scores_for_group(p, "uniform-grads"; metric=metric)
        if isnothing(xs_std) || isnothing(xs_grads); println("d=$d $metric: missing"); continue; end
        _, _, _, s1, e1, _ = compute_convergence_slope(xs_std,   ys_std)
        _, _, _, s2, e2, _ = compute_convergence_slope(xs_grads, ys_grads)
        gap = compute_logratio_gap_for_problem(p; metric=metric)
        println("d=$d $metric:  std=[$s1,$e1]  grads=[$s2,$e2]  gap=$gap")
    end
end

println("\n=== Regenerating plots ===")
plot_slopes_vs_dimension(; metric=:tv,          save_plot=true)
plot_slopes_vs_dimension(; metric=:convergence,  save_plot=true)
plot_gaps_vs_dimension_uniform(; metric=:tv,         save_plot=true)
plot_gaps_vs_dimension_uniform(; metric=:convergence, save_plot=true)
println("Done.")
