@show ARGS

# ARGS are:
# #1: problem name (matches a subtype of `AbstractProblem`)
# #2: run name (describes the used BOSIP setup)
# #3: run index
# #4: metric type name
# #5: estimator function
# #6: score_suffix (optional, e.g. "_norm"; default "")
# #7: model_type (optional, e.g. "nongp"; default "" = use loaded model)
const problem_name = ARGS[1]
const run_name = ARGS[2]
const run_idx = parse(Int, ARGS[3])
const metric_name = ARGS[4]
const estimator_name = ARGS[5]
const score_suffix = length(ARGS) >= 6 ? ARGS[6] : ""
const model_type = length(ARGS) >= 7 ? ARGS[7] : ""

### Calculate the scores
include("../src/calculate_score.jl")

const problem = getfield(Main, Symbol(problem_name))()
const metricT = getfield(Main, Symbol(metric_name))
const estimator = getfield(BOSIP, Symbol(estimator_name))

const model_override = if model_type == "nongp"
    NonstationaryGP(;
        mean = prior_mean(problem),
        lengthscale_model = BOSS.default_lengthscale_model(domain(problem).bounds, y_dim(problem)),
        amplitude_model = get_amplitude_priors(problem),
        noise_std_model = get_noise_std_priors(problem),
    )
else
    nothing
end

calculate_scores(problem, estimator, metricT, run_name, run_idx; save_score=true, score_suffix, model_override)
