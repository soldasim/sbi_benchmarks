# Find a representative "true parameter" x* for each opt-function problem.
# For each problem, x* satisfies f(x*) ≈ z_obs.
# Run from an interactive Julia session after include("src/main.jl").

using ForwardDiff, Printf, LinearAlgebra, Random

function find_true_params(problem; n_starts=200, seed=42)
    rng = Random.MersenneTwister(seed)
    f = true_f(problem)
    dom = domain(problem)
    lb, ub = dom.bounds
    z_target = prior_mean(problem)[1]   # works for both NormalLikelihood and CustomLikelihood (proxy) problems
    d = length(lb)

    function obj(x)
        (f(clamp.(x, lb, ub))[1] - z_target)^2
    end

    best_x = (lb .+ ub) ./ 2
    best_val = obj(best_x)

    for _ in 1:n_starts
        x = lb .+ rand(rng, d) .* (ub .- lb)

        # Gradient descent with backtracking line search.
        # Falls back to finite differences for non-smooth functions (abs etc.).
        for _ in 1:3000
            g = try
                ForwardDiff.gradient(obj, x)
            catch
                fd_eps = 1e-5
                [(obj(x .+ fd_eps .* (1:d .== j)) - obj(x)) / fd_eps for j in 1:d]
            end
            norm_g = norm(g)
            norm_g < 1e-12 && break

            step = 0.05 * minimum(ub .- lb)
            val0 = obj(x)
            for _ in 1:40
                x_new = clamp.(x .- step .* g ./ norm_g, lb, ub)
                if obj(x_new) < val0
                    x = x_new
                    break
                end
                step *= 0.5
            end
        end

        x = clamp.(x, lb, ub)
        val = obj(x)
        if val < best_val
            best_val = val
            best_x = copy(x)
        end
    end

    return best_x, sqrt(best_val)
end

problems = [
    RosenbrockProblem(x_dim=2),
    StyblinskiTangProblem(x_dim=2),
    MichalewiczProblem(x_dim=2),
    AckleyProblem(x_dim=2),
    AlpineProblem(x_dim=2),
    ExpandedSchafferF6Problem(x_dim=2),
    ExpandedZakharovProblem(x_dim=2),
    GriewankProblem(x_dim=2),
    RastriginProblem(x_dim=2),
    SalomonProblem(x_dim=2),
    SchwefelProblem(x_dim=2),
    SphereProblem(x_dim=2),
    BealeProxyProblem(),
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProxyProblem(),
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
]

out = open("src/true_params_results.txt", "w")

@printf out "%-36s %-10s %-32s %-12s %s\n" "Problem" "z_obs" "x*" "f(x*)" "err"
println(out, "-"^100)

for problem in problems
    name = get_name(problem)
    z_obs_val = prior_mean(problem)[1]
    x_true, err = find_true_params(problem)
    f_val = true_f(problem)(x_true)[1]
    x_str = "[" * join([@sprintf("%.6f", xi) for xi in x_true], ", ") * "]"
    @printf "%-36s %-10.4f %-32s %-12.6f %.2e\n" name z_obs_val x_str f_val err  # also print to REPL for monitoring
    @printf out "%-36s %-10.4f %-32s %-12.6f %.2e\n" name z_obs_val x_str f_val err
    flush(out)
end

println(out, "\nDone.")
close(out)
println("\nDone. Results written to src/true_params_results.txt")
