# ASSUMES pwd == sbi_benhmarks

include("../src/main.jl")

function queue_jobs(problem::AbstractProblem)
    run_name = "ig" # the name used for storing data from this run
    problem_name = string(typeof(problem))

    start_files = Glob.glob(data_dir(problem) * "/start_*.jld2")
    @info "Running $(length(start_files)) runs of the $(typeof(problem)) ..."
    
    for start_file in start_files
        m = match(r"start_(\d+)\.jld2$", start_file)
        run_idx = parse(Int, m.captures[1])
        # data = load(start_file, "data")

        # main(; run_name, save_data=true, data, run_idx)
        Base.run(`sbatch -p cpulong cluster_scripts/run.sh $run_name $problem_name $run_idx`)
    end

    nothing
end
