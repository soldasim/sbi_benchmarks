### Inspection script for TV metric instabilities in CubicProblem high-dimensional MaxVar runs.
### Run in an interactive Julia session (on a compute node via tmux):
###   include("inspect_cubic_instability.jl")

using JLD2
using Statistics
using CairoMakie

BASE = "/home/soldasim/repos/bosip_benchmarks"
DATA = BASE * "/data-convergence4"

dims = [4, 5, 6]
n_runs = 5
run_name = "maxvar"

# -------- Helper --------

function load_tv(dim, idx)
    dir = "$DATA/MultidimProblem{CubicProblem}$(dim)"
    f = "$dir/$(run_name)_$(idx)_TVmetric.jld2"
    d = load(f)
    return d["score"]  # Vector{Float64} or Matrix (1 x iters)
end

function load_data(dim, idx)
    dir = "$DATA/MultidimProblem{CubicProblem}$(dim)"
    f = "$dir/$(run_name)_$(idx)_data.jld2"
    d = load(f)
    X, Y = d["data"]
    return X, Y
end

function load_problem(dim, idx)
    dir = "$DATA/MultidimProblem{CubicProblem}$(dim)"
    f = "$dir/$(run_name)_$(idx)_problem.jld2"
    d = load(f)
    return d["problem"]
end

# -------- 1. Print TV score summaries --------

out_io = open(BASE * "/inspect_cubic_output.txt", "w")

function tee(s)
    println(s)
    println(out_io, s)
end

tee("\n===== TV score summaries =====")
all_tv = Dict()
for dim in dims
    tee("\n--- CubicProblem $(dim)D ---")
    for idx in 1:n_runs
        tv = load_tv(dim, idx)
        tv_vec = tv isa Matrix ? vec(tv) : tv
        all_tv[(dim, idx)] = tv_vec
        n = length(tv_vec)
        first_half = tv_vec[1:div(n,2)]
        second_half = tv_vec[div(n,2)+1:end]
        tee("  run $idx: n_iters=$(n), mean_1st=$(round(mean(first_half),digits=3)), mean_2nd=$(round(mean(second_half),digits=3)), max=$(round(maximum(tv_vec),digits=3)), final=$(round(tv_vec[end],digits=3))")
        avg = mean(tv_vec)
        spikes = findall(x -> x > 1.5 * avg, tv_vec)
        if !isempty(spikes)
            tee("    spikes at iters: $(spikes) (values: $(round.(tv_vec[spikes], digits=3)))")
        end
    end
end

# -------- 2. TV score plots --------

fig = Figure(size=(1000, 800))
for (i, dim) in enumerate(dims)
    ax = Axis(fig[i, 1], title="CubicProblem $(dim)D — MaxVar TV score", xlabel="iteration", ylabel="TV")
    for idx in 1:n_runs
        tv = load_tv(dim, idx)
        tv_vec = tv isa Matrix ? vec(tv) : tv
        lines!(ax, tv_vec, label="run $idx")
    end
    axislegend(ax, position=:rt)
end
save(BASE * "/inspect_cubic_tv.png", fig)
tee("\nSaved TV plot: $(BASE)/inspect_cubic_tv.png")

# -------- 3. Data point distribution --------

tee("\n===== Data point statistics =====")
for dim in dims
    tee("\n--- CubicProblem $(dim)D ---")
    for idx in 1:n_runs
        X, Y = load_data(dim, idx)
        n_pts = size(X, 2)
        per_dim_std = [std(X[d, :]) for d in 1:dim]
        per_dim_mean = [mean(X[d, :]) for d in 1:dim]
        tee("  run $idx: n_data=$(n_pts), per-dim std=$(round.(per_dim_std, digits=2)), per-dim mean=$(round.(per_dim_mean, digits=2))")
    end
end

# -------- 4. GP hyperparameters from saved problem --------

tee("\n===== GP hyperparameters (final state) =====")
for dim in dims
    tee("\n--- CubicProblem $(dim)D ---")
    for idx in 1:n_runs
        try
            p = load_problem(dim, idx)
            boss_problem = p.problem
            model = boss_problem.model
            if hasproperty(model, :params)
                params = model.params
                tee("  run $idx: model.params = $params")
            elseif hasproperty(boss_problem, :model_params)
                params = boss_problem.model_params
                tee("  run $idx: model_params = $params")
            else
                tee("  run $idx: model type = $(typeof(model))")
                tee("    boss_problem fields = $(fieldnames(typeof(boss_problem)))")
                tee("    model fields = $(fieldnames(typeof(model)))")
                if hasproperty(model, :length_scale)
                    tee("    length_scale = $(model.length_scale)")
                end
                if hasproperty(boss_problem, :data)
                    tee("    n_data = $(size(boss_problem.data.X, 2))")
                end
            end
        catch e
            tee("  run $idx: ERROR loading problem — $e")
        end
    end
end

# -------- 5. Correlate TV spike iterations with data count --------

tee("\n===== Checking if spikes correlate with late data points =====")
for dim in dims
    tee("\n--- CubicProblem $(dim)D ---")
    for idx in 1:n_runs
        tv_vec = all_tv[(dim, idx)]
        n = length(tv_vec)
        late_window = tv_vec[max(1, n-20):end]
        early_window = tv_vec[1:min(20, n)]
        ratio = mean(late_window) / mean(early_window)
        tee("  run $idx: late/early TV ratio = $(round(ratio, digits=2)) (early_mean=$(round(mean(early_window),digits=3)), late_mean=$(round(mean(late_window),digits=3)))")
    end
end

close(out_io)
tee("\nDone. Output saved to $(BASE)/inspect_cubic_output.txt")
