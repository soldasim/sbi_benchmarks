include("../src/main.jl")

@show ARGS

const run_name = ARGS[1]
const problem = getfield(Main, Symbol(ARGS[2]))()
const run_idx = parse(Int, ARGS[3])

const start_file = data_dir(problem) * "/start_$(run_idx).jld2"
const data = load(start_file, "data")

main(; run_name, save_data=true, metric=true, data, run_idx)
