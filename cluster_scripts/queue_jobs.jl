# ASSUMES pwd == sbi_benhmarks

include("../src/main.jl")

function queue_jobs(problem::AbstractProblem, run_name::String;
    selected_runs = nothing,
    continued = false,
    iters = 100,
    noise = "nothing",
)
    pname = get_name(problem)

    start_files = Glob.glob(starts_dir(problem) * "/start_*.jld2")
    n_runs = isnothing(selected_runs) ? length(start_files) : length(selected_runs)
    @info "Running $(n_runs) runs of the $(pname) ..."

    for start_file in start_files
        m = match(r"start_(\d+)\.jld2$", start_file)
        run_idx = parse(Int, m.captures[1])
        # data = load(start_file, "data")

        if !isnothing(selected_runs)
            (run_idx in selected_runs) || continue
        end

        # main(; run_name, save_data=true, data, run_idx)
        @info "Queuing run: problem:\"$(pname)\", run_name:\"$(run_name)\", run_idx:\"$(run_idx)\""
        # TODO --mem (the code was failing with the `SimpleProblem` with the default 4G memory)
        job_name = "$(pname)_$(run_name)_$(run_idx)"
        job_name = continued ? job_name * "_cont" : job_name
        cont = continued ? 1 : 0
        device = "cpu" # TODO "cpu"

        endswith(run_name, "noise") || @assert (noise == "nothing") # sanity check
        Base.run(`sbatch -p $device --mem=12G --job-name=$job_name cluster_scripts/run.sh $pname $run_name $run_idx $cont $iters $noise`)
    end

    nothing
end

function queue_a_lot()
    for d in 1:6
        queue_jobs(MeanGauss(; x_dim=d), "standard-warm"; selected_runs=collect(1:5), iters=1000)
        queue_jobs(MeanGauss(; x_dim=d), "grads-warm"; selected_runs=collect(1:5), iters=1000)
    end
    for d in 1:3
        queue_jobs(MultidimProblem(ABProblem(), d), "standard-warm"; selected_runs=collect(1:5), iters=1000)
        queue_jobs(MultidimProblem(ABProblem(), d), "grads-warm"; selected_runs=collect(1:5), iters=1000)
    end
end
