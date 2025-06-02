
function plot_results(problem::AbstractProblem)
    dir = data_dir(problem)

    files = sort(Glob.glob(joinpath(dir, "*.jld2")))
    scores_by_group = Dict{String, Vector{Vector{Float64}}}()

    for file in files
        fname = split(basename(file), ".")[1]
        group = split(fname, "_")[1]
        data = load(file)
        if haskey(data, "score")
            if !haskey(scores_by_group, group)
                scores_by_group[group] = Vector{Vector{Float64}}()
            end
            push!(scores_by_group[group], data["score"])
        end
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Iteration", ylabel="Median Score", title="Median Scores by Group")

    for (group, scores) in scores_by_group
        # scores is a Vector of score histories (each is a Vector)
        # Pad with NaN to equal length if needed
        maxlen = maximum(length.(scores))
        padded = [vcat(s, fill(NaN, maxlen - length(s))) for s in scores]
        arr = reduce(hcat, padded)
        # Compute statistics ignoring NaNs
        median_scores = mapslices(median∘skipmissing, arr; dims=2)[:]
        q25 = mapslices(x -> quantile(skipmissing(x), 0.25), arr; dims=2)[:]
        q75 = mapslices(x -> quantile(skipmissing(x), 0.75), arr; dims=2)[:]
        # Plot median line
        lines!(ax, 1:maxlen, median_scores, label=group)
        # Plot quantile band
        band!(ax, 1:maxlen, q25, q75)
    end

    axislegend(ax)
    fig
end
