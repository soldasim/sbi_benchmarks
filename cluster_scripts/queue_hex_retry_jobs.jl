# Resubmit the 31 failed continuation jobs after two fixes:
# 1. Assertion relaxed to data_count > 3 (was 3 + 100)
# 2. MetricCallback wraps _calc_score in try-catch (DomainError no longer crashes run)
# Run from the repo root: julia --project=src cluster_scripts/queue_hex_retry_jobs.jl

include("../src/main.jl")
include("queue_jobs.jl")

T = "24:00:00"

# MaxVar failures (DomainError in TV metric callback)
queue_jobs(HexObsProblem(GoldsteinPriceProxyProblem()), "maxvar";
    selected_runs=[2], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(HimmelblauProblem()), "maxvar";
    selected_runs=[1, 2, 3, 4], continued=true, iters=200, time=T)

# EIV failures (assertion or DomainError)
queue_jobs(HexObsProblem(RosenbrockProblem(; x_dim=2)), "eiv";
    selected_runs=[3], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(StyblinskiTangProblem(; x_dim=2)), "eiv";
    selected_runs=[2], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(GriewankProblem(; x_dim=2)), "eiv";
    selected_runs=[4], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(SchwefelProblem(; x_dim=2)), "eiv";
    selected_runs=[1, 2, 3, 4, 5], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(SphereProblem(; x_dim=2)), "eiv";
    selected_runs=[1, 3, 4], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(BealeProxyProblem()), "eiv";
    selected_runs=[1, 2, 3, 4, 5], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(GoldsteinPriceProxyProblem()), "eiv";
    selected_runs=[1, 2, 3, 4, 5], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(HimmelblauProblem()), "eiv";
    selected_runs=[1, 4], continued=true, iters=200, time=T)
queue_jobs(HexObsProblem(SchafferN2Problem()), "eiv";
    selected_runs=[1, 3, 5], continued=true, iters=200, time=T)

@info "Done queuing retry jobs."
