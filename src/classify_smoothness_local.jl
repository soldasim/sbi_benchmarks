"""
Local (near-truth) variant of classify_smoothness.jl's 5 metrics.

Round 1 of the metric search (2026-08-03): the original metrics are computed
from a DOMAIN-UNIFORM LHC sample, but BOSIP's active learning concentrates
observations near the true posterior mode — a surface that is badly-scaled
far from that region but fine near it (or vice versa) would score high/low on
the global metric while the advanced surrogate's actual benefit is determined
by local behavior only. This variant restricts sampling to a small box around
`true_params(problem)` (half-width = LOCAL_FRAC × domain width per dimension,
clipped to the domain bounds) — a cheap, robust stand-in for "where BO
actually looks" that avoids the low-effective-sample-size problems of naively
importance-weighting a domain-uniform sample for concentrated posteriors.

Does NOT modify classify_smoothness.csv or any BOSIP experiment data — writes
only to plots/classify_smoothness_local.csv.

Must be run after:
    include("src/main.jl")
    include("src/classify_smoothness.jl")
so that the problem lists and helper functions (_cv, _skewness,
_yj_lambda_dev, _heterosced, _sm_knn_indices, _sm_lhc, _sm_display_name,
LHC_SIZE) are in scope.

Usage:
    include("src/classify_smoothness_local.jl")
    run_classify_smoothness_local()
"""

const LOCAL_FRAC = 0.08   # local-box half-width as a fraction of domain width per dim

# Not every problem defines true_params (e.g. ABProblem/SimpleProblem/BananaProblem/
# BimodalProblem have non-identifiable/symmetric posteriors with no single "true"
# point). Fallback: cheaply search for a high-likelihood point by minimizing
# ||f(x) - prior_mean(problem)|| over an LHC sample (prior_mean is set to
# maximize the likelihood per the AbstractProblem convention, e.g. = z_obs).
function _approx_true_params(problem, f, lb, ub)
    Xs = _sm_lhc(lb, ub, 500)
    target = prior_mean(problem)
    best_i, best_d = 1, Inf
    for i in 1:500
        d = sum((f(@view Xs[:, i]) .- target) .^ 2)
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return Xs[:, best_i]
end

function _sm_local_box(problem, f)
    lb, ub = domain(problem).bounds
    xc = try
        true_params(problem)
    catch e
        e isa MethodError || rethrow()
        _approx_true_params(problem, f, lb, ub)
    end
    hw = LOCAL_FRAC .* (ub .- lb)
    loc_lb = max.(lb, xc .- hw)
    loc_ub = min.(ub, xc .+ hw)
    return loc_lb, loc_ub
end

function _analyse_smoothness_local(problem, name)
    println("Analysing $name  (local box, LHC_SIZE=$LHC_SIZE)")
    f  = true_f(problem)
    isnothing(f) && error("$name has no true_f defined — cannot evaluate proxy surface.")
    loc_lb, loc_ub = _sm_local_box(problem, f)
    dx = length(loc_lb)

    X = _sm_lhc(loc_lb, loc_ub, LHC_SIZE)

    y1 = f(@view X[:, 1])
    dy = length(y1)
    Y  = Matrix{Float64}(undef, dy, LHC_SIZE)
    Y[:, 1] = y1
    G  = Array{Float64}(undef, dy, LHC_SIZE)
    G[:, 1] = [norm(@view ForwardDiff.jacobian(f, X[:, 1])[j, :]) for j in 1:dy]

    Threads.@threads for i in 2:LHC_SIZE
        xi = X[:, i]
        Y[:, i] = f(xi)
        J = ForwardDiff.jacobian(f, xi)
        for j in 1:dy
            G[j, i] = norm(@view J[j, :])
        end
    end

    cv_value = maximum(_cv(@view Y[j, :]) for j in 1:dy)
    cv_grad  = maximum(_cv(@view G[j, :]) for j in 1:dy)
    yj_dev     = maximum(_yj_lambda_dev(@view Y[j, :]) for j in 1:dy)
    skewness   = maximum(abs(_skewness(@view Y[j, :])) for j in 1:dy)
    knn_idxs   = _sm_knn_indices(X; k=20)
    heterosced = maximum(abs(_heterosced(knn_idxs, @view Y[j, :])) for j in 1:dy)

    kurtosis         = maximum(_kurtosis(@view Y[j, :]) for j in 1:dy)
    hessian_cond     = _hessian_anisotropy(f, X, dy)
    lipschitz        = maximum(_lipschitz(X, @view Y[j, :]) for j in 1:dy)
    path_roughness   = _path_roughness(f, loc_lb, loc_ub, dy)
    output_redundancy = _output_redundancy(Y)

    return (name=name, dx=dx, dy=dy, cv_value=cv_value, cv_grad=cv_grad,
            yj_dev=yj_dev, skewness=skewness, heterosced=heterosced,
            kurtosis=kurtosis, hessian_cond=hessian_cond, lipschitz=lipschitz,
            path_roughness=path_roughness, output_redundancy=output_redundancy)
end

_sm_local_analyse_problem(p) = _analyse_smoothness_local(p, _sm_display_name(p))

function run_classify_smoothness_local()
    results = map(_sm_local_analyse_problem, _SM_ALL_PROBLEMS)

    println()
    for r in results
        println("$(r.name): cv_value=$(round(r.cv_value,digits=3)) cv_grad=$(round(r.cv_grad,digits=3)) " *
                "yj_dev=$(round(r.yj_dev,digits=3)) skewness=$(round(r.skewness,digits=3)) " *
                "heterosced=$(round(r.heterosced,digits=3)) kurtosis=$(round(r.kurtosis,digits=3)) " *
                "hessian_cond=$(round(r.hessian_cond,digits=3)) lipschitz=$(round(r.lipschitz,digits=3)) " *
                "path_roughness=$(round(r.path_roughness,digits=3)) output_redundancy=$(round(r.output_redundancy,digits=3))")
    end

    mkpath("plots")
    open("plots/classify_smoothness_local.csv", "w") do io
        println(io, "problem,dx,dy,cv_value,cv_grad,yj_dev,skewness,heterosced,kurtosis,hessian_cond,lipschitz,path_roughness,output_redundancy")
        for r in results
            println(io, "$(r.name),$(r.dx),$(r.dy),$(r.cv_value),$(r.cv_grad),$(r.yj_dev),$(r.skewness),$(r.heterosced)," *
                        "$(r.kurtosis),$(r.hessian_cond),$(r.lipschitz),$(r.path_roughness),$(r.output_redundancy)")
        end
    end
    println("\nSaved → plots/classify_smoothness_local.csv")
    return results
end
