include("main.jl")

function plot_results(problem::AbstractProblem; save_plot=false)
    ### setings
    dir = data_dir(problem)
    plotted_groups = ["standard", "absval", "ig2"]
    ###

    files = sort(Glob.glob(joinpath(dir, "*.jld2")))
    scores_by_group = Dict{String, Vector{Vector{Float64}}}()

    for file in files
        fname = split(basename(file), ".")[1]
        group = split(fname, "_")[1]
        
        startswith(fname, "start") && continue
        endswith(fname, "extras") && continue

        data = load(file)
        @assert haskey(data, "score")
        if !haskey(scores_by_group, group)
            scores_by_group[group] = Vector{Vector{Float64}}()
        end
        push!(scores_by_group[group], data["score"])
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Iteration", ylabel="Median Score", title="Median Scores by Group", yscale=log)

    colors = Makie.wong_colors()  # or use any preferred color palette
    group_names = collect(keys(scores_by_group))
    color_map = Dict(group => colors[mod1(i, length(colors))] for (i, group) in enumerate(group_names))

    for (group, scores) in scores_by_group
        group in plotted_groups || continue

        color = color_map[group]
        # scores is a Vector of score histories (each is a Vector)
        # Pad with NaN to equal length if needed
        maxlen = maximum(length.(scores))
        padded = [vcat(s, fill(NaN, maxlen - length(s))) for s in scores]
        arr = reduce(hcat, padded)
        xs = 0:maxlen-1

        # Compute statistics ignoring NaNs
        median_scores = mapslices(median∘skipmissing, arr; dims=2)[:]
        q25 = mapslices(x -> quantile(skipmissing(x), 0.25), arr; dims=2)[:]
        q75 = mapslices(x -> quantile(skipmissing(x), 0.75), arr; dims=2)[:]
        # Plot median line
        lines!(ax, xs, median_scores, label=group, color=color)
        # Plot quantile band with alpha
        band!(ax, xs, q25, q75; color=color, alpha=0.6)

        q10 = mapslices(x -> quantile(skipmissing(x), 0.1), arr; dims=2)[:]
        q90 = mapslices(x -> quantile(skipmissing(x), 0.9), arr; dims=2)[:]
        lines!(ax, xs, q10; color=color, linestyle=:dot, linewidth=1)
        lines!(ax, xs, q90; color=color, linestyle=:dot, linewidth=1)
    end

    axislegend(ax)

    save_plot && save(plot_dir() * "/" * string(typeof(problem)) * ".png", fig)
    fig
end
