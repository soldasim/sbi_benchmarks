# ASSUMES pwd == sbi_benhmarks

include("../src/main.jl")

function queue_jobs(problem::AbstractProblem, run_name::String)
    problem_name = string(typeof(problem))

    start_files = Glob.glob(starts_dir(problem) * "/start_*.jld2")
    @info "Running $(length(start_files)) runs of the $(typeof(problem)) ..."
    
    for start_file in start_files
        m = match(r"start_(\d+)\.jld2$", start_file)
        run_idx = parse(Int, m.captures[1])
        # data = load(start_file, "data")

        # main(; run_name, save_data=true, data, run_idx)
        @info "Queuing run: problem:\"$(problem_name)\", run_name:\"$(run_name)\", run_idx:\"$(run_idx)\""
        # TODO --mem (the code was failing with the `SimpleProblem` with the default 4G memory)
        Base.run(`sbatch -p cpulong --mem=16G cluster_scripts/run.sh $problem_name $run_name $run_idx`)
    end

    nothing
end
