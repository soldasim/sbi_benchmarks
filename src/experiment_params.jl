
# Number of grid points used for the TV metric MC integration.
_grid_size(problem::AbstractProblem) = 20_000

# Number of x-samples used for integral-based acquisition functions (EIV, IMMD, IMIQR, ...).
# Capped at 2k to keep runtimes feasible at higher dimensions (~10 min/iter at d=5).
_acq_samples(problem::AbstractProblem) = min(2 * 10^x_dim(problem), 2_000)
