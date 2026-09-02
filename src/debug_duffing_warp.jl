## Debug script: confirm root cause of DomainError in MetricCallback for DuffingProblem.

using JLD2, Distributions, BOSS, BOSIP

bosip = load("data-warpedgp/DuffingProblem/warpedgp-maxvar_1_problem.jld2")["problem"]
println("Loaded BosipProblem. Data size: ", size(bosip.problem.data.X))

params_inner = bosip.problem.params isa BOSS.MAPParams ? bosip.problem.params.params : bosip.problem.params
wp_all = params_inner.warp[1]
println("Warp params: λ=$(wp_all[1])  a=$(wp_all[2])  b=$(wp_all[3])")

model_post = BOSS.model_posterior(bosip.problem)
slice_post = model_post.slices[1]

grid = load("data-bosip-norm/DuffingProblem/grid/posterior_grid.jld2")["xs"]
approx_logpost = BOSIP.log_posterior_mean(bosip)

function scan_grid(approx_logpost, grid)
    n_ok = 0; n_nan = 0; n_err = 0; fail_idx = nothing
    for i in axes(grid, 2)
        x = grid[:, i]
        try
            val = approx_logpost(x)
            (isnan(val) || isinf(val)) ? (n_nan += 1; isnothing(fail_idx) && (fail_idx = i)) : (n_ok += 1)
        catch e
            n_err += 1; isnothing(fail_idx) && (fail_idx = i)
        end
    end
    println("Grid scan: ok=$n_ok, NaN/Inf=$n_nan, DomainError=$n_err  (total=$(size(grid,2)))")
    return fail_idx
end

fail_idx = scan_grid(approx_logpost, grid)

if !isnothing(fail_idx)
    x_fail = grid[:, fail_idx]
    println("\n--- Deep dive at grid[$fail_idx] = $x_fail ---")

    # Get latent GP posterior
    m_v, s2_v = BOSS.mean_and_var(slice_post.post_gp, hcat(x_fail); obsdim=2) |> t -> (mean(t), var(t))
    m = first(m_v); s2 = first(s2_v)
    println("  Latent GP: m=$m  σ²=$s2")
    s = sqrt(max(s2, 0.0))

    # Gauss-Hermite quadrature back-transform
    nodes = slice_post.nodes; wts = slice_post.weights
    wp = slice_post.warp_params; w = slice_post.warping
    latent_pts = @. m + sqrt(2) * s * nodes
    obs_vals = BOSS.warp_inverse.(Ref(w), Ref(wp), latent_pts)
    println("  GH latent range: [$(minimum(latent_pts)), $(maximum(latent_pts))]")
    println("  GH obs range:    [$(minimum(obs_vals)), $(maximum(obs_vals))]")
    println("  Any Inf in obs_vals? ", any(isinf.(obs_vals)))

    norm_gh = inv(sqrt(π))
    Ey  = norm_gh * sum(wts .* obs_vals)
    Ey2 = norm_gh * sum(wts .* obs_vals .^ 2)
    var_y = Ey2 - Ey^2
    println("  E[y]=$(Ey)  E[y²]=$(Ey2)  Var[y]=$(var_y)")
    println("  max(Var,0)=$(max(var_y, 0.0))  → NaN because max(NaN,0)=NaN in Julia")

    # Show which individual GH nodes produce Inf
    println("\n  GH node details (extreme values only):")
    for (i, (n, o)) in enumerate(zip(latent_pts, obs_vals))
        (isinf(o) || abs(o) > 1e6) && println("    node[$i]: latent=$n → obs=$o")
    end

    # SinhArcsinh and YJ inverse step by step for the worst node
    worst = argmax(abs.(obs_vals))
    lat_worst = latent_pts[worst]
    println("\n  Worst node: latent=$lat_worst")

    # SinhArcsinh inverse: applied first (second warp in reverse order)
    a_sa = wp[2]; b_sa = wp[3]
    δ_sa = sinh((asinh(lat_worst) + a_sa) / b_sa)
    println("  After SinhArcsinh⁻¹(δ=$(lat_worst); a=$a_sa, b=$b_sa): $δ_sa")
    println("    asinh(δ)=$(asinh(lat_worst))  (asinh+a)/b=$((asinh(lat_worst)+a_sa)/b_sa)")

    # YJ inverse: applied second
    λ = wp[1]
    if δ_sa >= 0
        base = λ * δ_sa + 1
        δ_yj = abs(λ) < BOSS.YJ_TOL ? exp(δ_sa) - 1 : max(base, 0.0)^(1/λ) - 1
    else
        base = 1 - (2 - λ) * δ_sa
        δ_yj = abs(λ - 2) < BOSS.YJ_TOL ? 1 - exp(-δ_sa) : 1 - max(base, 0.0)^(1/(2-λ))
    end
    println("  After YJ⁻¹(λ=$λ): $δ_yj")
end
