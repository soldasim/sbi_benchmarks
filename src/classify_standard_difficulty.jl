"""
Standard GP's own difficulty (2026-08-04): median and spread of Standard/MaxVar's
OWN full-length final-iteration log-TV score, independent of any fair-comparison
truncation. Tests the "headroom" hypothesis directly: maybe WarpedGP/NonstatGP's
apparent benefit is really about how well/badly the BASELINE does on a problem,
not about any particular surface-shape metric.

  - `standard_median`  — median over runs of each run's own last valid log-TV
    (same "last-valid-point" definition as compute_acq_scores_final.jl's
    `_last_valid_log_tv`, reproduced standalone here).
  - `standard_spread`  — std across runs of that same per-run score. A high
    spread suggests an unstable/noisy problem where any comparison margin
    might be dominated by noise rather than a genuine method difference.

  - `standard_conv_slope` — median (over runs) OLS slope of log(TV) vs.
    iteration index over each run's own valid trajectory. More negative =
    faster-converging baseline. Tests whether WarpedGP/NonstatGP's benefit
    concentrates on slow-converging problems. (Originally planned to also
    extract fitted lengthscale-anisotropy/noise-amplitude-ratio from Standard's
    fitted GP hyperparameters, but `MAPParams{GaussianProcess}` JLD2 files hit
    a genuine version-compatibility wall — a field was renamed in BOSS.jl since
    these files were written, and JLD2 can't reconstruct the object into the
    strongly-typed parent field. WarpedGP's params didn't hit this. Not fixed —
    would require reverse-engineering the exact struct migration; out of scope
    here. This slope metric only needs the TV score vectors, which always load
    fine, so it sidesteps the issue entirely.)

Must be run after include("src/main.jl"); include("src/classify_smoothness.jl")
(for the problem list / _sm_display_name). Read-only w.r.t. experiment data —
writes only plots/classify_standard_difficulty.csv.

Usage:
    include("src/classify_standard_difficulty.jl")
    run_classify_standard_difficulty()
"""

using JLD2
using Statistics: mean, median, std

const _SD_BIP_NAMES = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                            "SIRProblem", "DuffingProblem", "DiffusionProblem10",
                            "DuffingProblem5", "DiffusionProblem5D"])

function _std_last_valid_log_tv(prob_dir, run_name, idx)
    tv_path = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
    isfile(tv_path) || return nothing
    try
        score = load(tv_path, "score")
        i = findlast(v -> isfinite(v) && v > 0.0, score)
        i === nothing && return nothing
        return log(score[i])
    catch
        return nothing
    end
end

function _std_collect(prob_dir, run_name; n=20)
    scores = Float64[]
    for idx in 1:n
        v = _std_last_valid_log_tv(prob_dir, run_name, idx)
        v !== nothing && push!(scores, v)
    end
    return scores
end

# OLS slope of log(TV) vs iteration index over the run's own valid trajectory.
function _conv_slope(prob_dir, run_name, idx)
    tv_path = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
    isfile(tv_path) || return nothing
    try
        score = load(tv_path, "score")
        valid = findall(v -> isfinite(v) && v > 0.0, score)
        length(valid) < 5 && return nothing
        xs = Float64.(valid)
        ys = log.(score[valid])
        xb, yb = mean(xs), mean(ys)
        denom = sum((xs .- xb) .^ 2)
        denom < 1e-12 && return nothing
        return sum((xs .- xb) .* (ys .- yb)) / denom
    catch
        return nothing
    end
end

function _std_collect_slopes(prob_dir, run_name; n=20)
    slopes = Float64[]
    for idx in 1:n
        v = _conv_slope(prob_dir, run_name, idx)
        v !== nothing && push!(slopes, v)
    end
    return slopes
end

function _std_dir_and_key(name)
    if name in _SD_BIP_NAMES
        return joinpath("data-bosip-norm", name), "standard"
    else
        return joinpath("data-opt-functions", name * "_cross"), "maxvar"
    end
end

function run_classify_standard_difficulty()
    results = NamedTuple[]
    for p in _SM_ALL_PROBLEMS
        name = _sm_display_name(p)
        dir, key = _std_dir_and_key(name)
        if !isdir(dir)
            @warn "Missing $dir for $name"
            continue
        end
        scores = _std_collect(dir, key; n=20)
        if isempty(scores)
            @warn "No scores for $name"
            continue
        end
        med = median(scores)
        spr = length(scores) > 1 ? std(scores) : 0.0
        slopes = _std_collect_slopes(dir, key; n=20)
        slope_med = isempty(slopes) ? 0.0 : median(slopes)
        push!(results, (name=name, standard_median=med, standard_spread=spr, standard_conv_slope=slope_med, n=length(scores)))
        println("$name: standard_median=$(round(med,digits=3)) standard_spread=$(round(spr,digits=3)) standard_conv_slope=$(round(slope_med,digits=5)) (n=$(length(scores)))")
    end
    mkpath("plots")
    open("plots/classify_standard_difficulty.csv", "w") do io
        println(io, "problem,standard_median,standard_spread,standard_conv_slope,n")
        for r in results
            println(io, "$(r.name),$(r.standard_median),$(r.standard_spread),$(r.standard_conv_slope),$(r.n)")
        end
    end
    println("\nSaved → plots/classify_standard_difficulty.csv")
    return results
end
