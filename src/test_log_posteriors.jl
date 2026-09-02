# Test that the proxy problem variants produce numerically identical true posteriors
# to their originals.
#
# The proxy problems use CustomLikelihood which inverts the proxy transformation
# (f = exp(δ) − C) and evaluates the original NormalLikelihood on f_raw. Combined
# with the same x_prior, this gives true_logpost(proxy, x) == true_logpost(orig, x)
# for all x up to floating-point rounding.
#
# What we test:
#   1. The proxy simulator output is finite everywhere on the grid.
#   2. The proxy log-posterior is numerically identical to the original (max |Δ| < tol).
#   3. The proxy simulator range is reported for sanity checking.

include("main.jl")

function test_proxy_problem(orig::AbstractProblem, proxy::AbstractProblem; grid_size=40, tol=1e-4)
    name_orig  = get_name(orig)
    name_proxy = get_name(proxy)
    println("\n── Testing $name_proxy vs $name_orig ──")

    lb, ub = domain(orig).bounds
    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect

    lp_orig  = true_logpost(orig)
    lp_proxy = true_logpost(proxy)
    f_proxy  = true_f(proxy)

    max_diff = 0.0
    sim_vals = Float64[]
    ok = true

    for x1 in xs1, x2 in xs2
        x = [x1, x2]

        y_proxy = f_proxy(x)[1]
        if !isfinite(y_proxy)
            println("  FAIL: f_proxy($x) = $y_proxy (not finite)")
            ok = false
            continue
        end
        push!(sim_vals, y_proxy)

        v_orig  = lp_orig(x)
        v_proxy = lp_proxy(x)

        if !isfinite(v_orig) || !isfinite(v_proxy)
            println("  FAIL: logpost not finite at $x: orig=$v_orig, proxy=$v_proxy")
            ok = false
            continue
        end

        diff = abs(v_orig - v_proxy)
        max_diff = max(max_diff, diff)
    end

    println("  Proxy simulator range:  [$(round(minimum(sim_vals),digits=4)), $(round(maximum(sim_vals),digits=4))]")
    println("  Max |logpost_orig − logpost_proxy|: $(max_diff)")

    if max_diff > tol
        println("  FAIL: max difference $max_diff exceeds tolerance $tol")
        ok = false
    else
        println("  OK: posteriors are numerically identical (tol=$tol)")
    end

    return ok
end

println("="^50)
println("Proxy posterior identity tests")
println("="^50)

all_pass = true
all_pass &= test_proxy_problem(BealeProblem(),          BealeProxyProblem())
all_pass &= test_proxy_problem(GoldsteinPriceProblem(), GoldsteinPriceProxyProblem())

println("\n" * "="^50)
println(all_pass ? "All tests passed." : "Some tests failed — see above.")
println("="^50)
