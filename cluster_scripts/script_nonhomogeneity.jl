@show ARGS

# ARGS are:
# #1: problem_name — reconstructible problem name (matches reconstruct_problem, may
#     include a "_cross" suffix), NOT the display name used in classify_smoothness.csv.
# #2: dim_idx — 1-based output dimension index (1..y_dim(problem))
const problem_name = ARGS[1]
const dim_idx = parse(Int, ARGS[2])

include("../src/main.jl")
include("../src/classify_smoothness.jl")
include("../src/classify_nonhomogeneity.jl")

const problem = reconstruct_problem(problem_name)

run_classify_nonhomogeneity_dim(problem, problem_name, dim_idx)

# Force exit to avoid hanging in finalizers from native-code libraries (PRIMA/Fortran).
exit()
