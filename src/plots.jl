###
### Plotting scripts for benchmark results
###

include("main.jl")
using Printf

axis_size() = (400, 300)

# Set larger font sizes for better readability
function set_theme_fonts!(; base_fontsize=20)
    set_theme!(
        fontsize = base_fontsize,
        Axis = (
            titlesize = base_fontsize + 2,
            xlabelsize = base_fontsize,
            ylabelsize = base_fontsize,
            xticklabelsize = base_fontsize - 2,
            yticklabelsize = base_fontsize - 2,
        ),
        Legend = (
            titlesize = base_fontsize,
            labelsize = base_fontsize - 1,
        ),
        Label = (
            textsize = base_fontsize,
        )
    )
end

function get_run_label(abbr::AbstractString)
    # group_labels = Dict([
    #     # "standard" => "GP - output - MaxVar",
    #     # "loglike" => "GP - loglike - MaxVar",
    #     # "loglike-imiqr" => "GP - loglike - IMIQR",
    #     # "eiv" => "GP - output - EIV",
    #     # "eiig" => "GP - output - IMMD",
    #     # "nongp" => "nonGP - output - MaxVar",
    #     # "tnp" => "TNP-D - output - MaxVar",
    #     # "bi" => "GP-BI - output - MaxVar",
    #     # "grads" => "grad-GP - output - MaxVar",
    #     # "grads-divr" => "grad-GP - output - dIVR",

    #     "standard" => "without gradients",
    #     "grads" => "with gradients",

    #     # TODO comment out
    #     # ### for estimator plot
    #     # "standard" => "GP - output - MaxVar - exp.",
    #     # "est" => "GP - output - MaxVar - MAP",
    #     # "loglike" => "GP - loglike - MaxVar - MAP",
    #     # "loglike-imiqr" => "GP - loglike - IMIQR - MAP",
    # ])
    # return get(group_labels, abbr, abbr)

    if occursin("uniform", abbr)
        return "uniform (random)"
    elseif abbr == "maxvar"
        return "MaxVar"
    elseif abbr == "eiv"
        return "EIV"
    elseif abbr == "immd"
        return "IMMD"
    elseif abbr == "eiig"
        return "IMMD (old)"
    elseif occursin("standard", abbr)
        return "without gradients"
    elseif occursin("grads", abbr)
        return "with gradients"
    else
        @warn "You should handle the label for group \"$abbr\"..."
        return abbr
    end
end

"""
    detect_linear_phase(xs::AbstractVector, ys::AbstractVector)

Detect the learning phase by finding the region with the best linear fit in log-log space,
balancing two objectives: 
1. Goodness of fit (constant slope, low residuals)
2. Region size (larger regions preferred)

Returns (init_end_idx, learning_end_idx, region_strength) where init_end_idx and learning_end_idx 
mark the start and end of the detected linear region, and region_strength is the fit score.
"""
function detect_linear_phase(xs::AbstractVector, ys::AbstractVector)
    @assert length(xs) == length(ys)
    
    n = length(xs)
    if n < 50
        return 1, n, 0.0
    end
    
    log_xs = log10.(xs)
    log_ys = log10.(ys)
    
    # Normalize deviations by the global y range (not region-specific) for fair comparison
    global_y_range = maximum(log_ys) - minimum(log_ys)
    
    # First pass: detect where the initial constant phase ends
    # Find the first point where deviation from start exceeds a threshold
    const_phase_threshold = 0.05 * global_y_range  # 10% of global y range
    initial_phase_end_idx = 1
    
    for i in 2:n
        # Distance from starting point
        deviation = abs(log_ys[i] - log_ys[1])
        if deviation > const_phase_threshold
            initial_phase_end_idx = i - 1
            break
        end
    end
    
    # If no clear constant phase was detected, initial_phase_end_idx stays at 1 (include all data)
    
    @debug "Detected initial constant phase ending at index $initial_phase_end_idx (out of $n)"
    
    # For each possible start point, find the best linear fit
    # We'll score each region by: (1 - normalized_rmse) * (region_fraction)
    # This balances fit quality with region size
    
    best_score = 0.0
    best_start_idx = initial_phase_end_idx
    best_end_idx = n
    
    # Minimum region size measured in non-log space (actual x values)
    x_range = xs[end] - xs[1]
    min_region_frac = 0.15  # 15% of total x-range
    min_x_region = min_region_frac * x_range
    
    for start_idx in initial_phase_end_idx:(n - 1)
        # Find the minimum end_idx satisfying the x-distance constraint
        min_end_idx_offset = findfirst(x -> x >= xs[start_idx] + min_x_region, xs[start_idx:end])
        if isnothing(min_end_idx_offset)
            continue  # No region large enough from this start point
        end
        min_end_idx = start_idx + min_end_idx_offset - 1
        
        # Try this starting point with various end points
        for end_idx in min_end_idx:n
            xs_region = log_xs[start_idx:end_idx]
            ys_region = log_ys[start_idx:end_idx]
            
            # Fit a line by connecting start and end points (chord fit)
            # This creates a reference line without free optimization
            b = (log_ys[end_idx] - log_ys[start_idx]) / (log_xs[end_idx] - log_xs[start_idx])
            a = log_ys[start_idx] - b * log_xs[start_idx]
            
            # Compute residuals
            y_pred = a .+ b .* xs_region
            residuals = ys_region .- y_pred
            
            # Use maximum deviation (L∞ norm) from the chord
            max_deviation = maximum(abs.(residuals))
            
            # Normalize max_deviation by the global range of y values (not region-specific)
            # This ensures fair comparison across regions with different y value ranges
            if global_y_range < 1e-10
                normalized_max_dev = 1.0
            else
                normalized_max_dev = max_deviation / global_y_range
            end
            
            # Score: balance linearity (fit_quality) with region size (region_frac)
            # Using 1 - normalized_max_dev as fit quality (0 to 1, higher is more linear)
            fit_quality = max(0, 1 - normalized_max_dev)  # 0 to 1, higher is better (more linear)
            
            # Measure region size in log-log space (not index space)
            # This ensures that regions of comparable length in log-space are penalized equally
            log_x_range = log_xs[end] - log_xs[1]
            log_region_range = log_xs[end_idx] - log_xs[start_idx]
            region_frac = log_region_range / log_x_range  # Fraction of data in log-space
            
            # Score: balance linearity with region size
            score = (fit_quality ^ 2) * region_frac
            
            if score > best_score
                best_score = score
                best_start_idx = start_idx
                best_end_idx = end_idx
            end
        end
    end
    
    return best_start_idx, best_end_idx, best_score
end

"""
    compute_convergence_slope(xs::AbstractVector, ys::AbstractVector)

Compute the convergence slope in log-log space for the linear phase.
Returns (slope, init_end_idx, learning_end_idx, init_x, learning_end_x, region_strength) tuple.
"""
function compute_convergence_slope(xs::AbstractVector, ys::AbstractVector)
    init_end_idx, learning_end_idx, region_strength = detect_linear_phase(xs, ys)
    
    # Get the linear phase region
    phase_xs = xs[init_end_idx:learning_end_idx]
    phase_ys = ys[init_end_idx:learning_end_idx]
    
    # Convert to log-log space for linear regression
    log_xs = log10.(phase_xs)
    log_ys = log10.(phase_ys)
    
    # Linear regression: log_y = slope * log_x + intercept
    n = length(log_xs)
    mean_x = mean(log_xs)
    mean_y = mean(log_ys)
    
    slope = sum((log_xs .- mean_x) .* (log_ys .- mean_y)) / sum((log_xs .- mean_x) .^ 2)
    
    return slope, init_end_idx, learning_end_idx, xs[init_end_idx], xs[learning_end_idx], region_strength
end

"""
    compute_constant_gap(xs::AbstractVector, ys_standard::AbstractVector, ys_grads::AbstractVector, 
                         phase_std::Tuple, phase_grads::Tuple)

Compute the constant gap between standard and gradient-augmented approaches over their overlapping linear phases.
The gap is computed as the difference in actual values (not log space).
    
phase_std and phase_grads should be tuples of (linear_start_x, linear_end_x, region_strength).
Returns the average gap (standard - grads) over the overlapping region, or nothing if overlap is empty.
"""
function compute_constant_gap(xs::AbstractVector, ys_standard::AbstractVector, ys_grads::AbstractVector, 
                               phase_std::Tuple, phase_grads::Tuple)
    # Extract phase boundaries
    start_std, end_std, _ = phase_std
    start_grads, end_grads, _ = phase_grads
    
    # Find overlap region
    overlap_start = max(start_std, start_grads)
    overlap_end = min(end_std, end_grads)
    
    # Check if there's a valid overlap
    if overlap_start >= overlap_end
        @warn "No overlap between linear phases: standard [$start_std, $end_std], grads [$start_grads, $end_grads]"
        return nothing
    end
    
    # Find indices for the overlap region
    indices_in_overlap = findall(x -> overlap_start <= x <= overlap_end, xs)
    
    if isempty(indices_in_overlap)
        @warn "No data points found in overlap region [$overlap_start, $overlap_end]"
        return nothing
    end
    
    # Compute average gap in actual values
    gaps = ys_standard[indices_in_overlap] .- ys_grads[indices_in_overlap]
    med_gap = median(gaps)
    
    return med_gap
end

### Create a grid of plots for the individual problems
### (Edit plotted runs in `plotted groups`.)
#
# ## Important Keywords
# - xscale
# - yscale
# - max_iters
function plot_results(; save_plot=false, base_fontsize=20, kwargs...)
    # Set theme with larger fonts
    set_theme_fonts!(base_fontsize=base_fontsize)
    
    ### problem grid
    # problems = [
    #     ABProblem()     SimpleProblem()     BananaProblem()     BimodalProblem()
    #     :legend         ProxySIRProblem()        DuffingProblem()    DiffusionProblem10()
    # ]
    # problems = [
    #     :legend         SIRProblem()         ProxySIRProblem()
    # ]
    # problems = [
    #     MultidimProblem(ABProblem(), 3)     MultidimProblem(SimpleProblem(), 3)     MultidimProblem(BananaProblem(), 3)     MultidimProblem(BimodalProblem(), 3)
    # ]

    # problems = [
    #     MultidimProblem(ABProblem(), 1)         MultidimProblem(ABProblem(), 2)         MultidimProblem(ABProblem(), 3)         MultidimProblem(ABProblem(), 4)         MultidimProblem(ABProblem(), 5)
    #     MultidimProblem(SimpleProblem(), 1)     MultidimProblem(SimpleProblem(), 2)     MultidimProblem(SimpleProblem(), 3)     MultidimProblem(SimpleProblem(), 4)     MultidimProblem(SimpleProblem(), 5)
    # ]
    problems = [
        :legend              MultidimProblem(ABProblem(), 1)              :nothing                            MultidimProblem(ABProblem(), 2)              :nothing                            MultidimProblem(ABProblem(), 3)
        MeanGauss(;x_dim=1)  MeanGauss(;x_dim=2)                          MeanGauss(;x_dim=3)                 MeanGauss(;x_dim=4)                          MeanGauss(;x_dim=5)                 MeanGauss(;x_dim=6)
        MultidimProblem(SquareProblem(), 1)  MultidimProblem(SquareProblem(), 2)  MultidimProblem(SquareProblem(), 3)  MultidimProblem(SquareProblem(), 4)  MultidimProblem(SquareProblem(), 5)  MultidimProblem(SquareProblem(), 6)
        MultidimProblem(SineProblem(), 1)    MultidimProblem(SineProblem(), 2)    MultidimProblem(SineProblem(), 3)    MultidimProblem(SineProblem(), 4)    MultidimProblem(SineProblem(), 5)    MultidimProblem(SineProblem(), 6)
        MultidimProblem(CubicProblem(), 1)   MultidimProblem(CubicProblem(), 2)   MultidimProblem(CubicProblem(), 3)   MultidimProblem(CubicProblem(), 4)   MultidimProblem(CubicProblem(), 5)   MultidimProblem(CubicProblem(), 6)
    ]

    nrows, ncols = size(problems)
    ax_width, ax_height = axis_size()
    fig = Figure(;
        size = (ax_width * ncols, ax_height * nrows),
    )

    # First pass: create all actual plots
    special_indices = []
    gaps_by_problem = Dict()  # Store gaps for each problem
    for idx in CartesianIndices(problems)
        if problems[idx] isa AbstractProblem
            ps = AbstractProblem[problems[idx]]
            ax, gap_result = plot_result_axis!(fig[idx.I...], ps; legend=false, kwargs...)
            # Store gap if available
            if !isnothing(gap_result)
                gaps_by_problem[ps[1]] = gap_result
            end
        elseif problems[idx] isa Tuple
            ps = AbstractProblem[problems[idx]...]
            ax, gap_result = plot_result_axis!(fig[idx.I...], ps; legend=false, kwargs...)
            # Store gap if available
            if !isnothing(gap_result)
                gaps_by_problem[ps[1]] = gap_result
            end
        else
            push!(special_indices, idx)
        end
    end

    # Save gap data for all problems
    for (problem, gap_result) in gaps_by_problem
        gap_value, phase_std, phase_grads, slope_std, slope_grad, ratio, metric = gap_result
        
        # Create gaps directory if it doesn't exist
        gaps_path = gaps_dir(problem)
        !isdir(gaps_path) && mkpath(gaps_path)
        
        # Save gaps data with proper symbol keys
        gaps_file = gaps_filepath(problem; metric=metric)
        jldsave(gaps_file;
            gap=gap_value,
            phase_standard=phase_std,
            phase_grads=phase_grads,
            slope_standard=slope_std,
            slope_grads=slope_grad,
            ratio=ratio,
            metric=string(metric)
        )
        @info "Saved gaps (metric=$metric) for $(get_name(problem)) to $gaps_file"
    end

    # Second pass: handle special elements (like :legend) after plots are created
    for idx in special_indices
        val = problems[idx]
        plot_result_special!(fig[idx.I...], fig, val; kwargs...)
    end

    # Set all columns and rows to the same size
    for col in 1:ncols
        colsize!(fig.layout, col, Relative(1 / ncols))
    end
    for row in 1:nrows
        rowsize!(fig.layout, row, Relative(1 / nrows))
    end

    if save_plot
        save(plot_dir() * "/all_problems.png", fig)
        save(plot_dir() * "/all_problems.pdf", fig)
    end
    return fig
end

### Create a single plot with the runs of the given problems
### (Edit plotted runs in `plotted groups`.)
plot_results(problem::AbstractProblem; kwargs...) = plot_results(AbstractProblem[problem]; kwargs...)
function plot_results(problems::AbstractVector; save_plot=false, base_fontsize=20, kwargs...)
    # Set theme with larger fonts
    set_theme_fonts!(base_fontsize=base_fontsize)
    
    fig = Figure()
    
    ax, gap_result = plot_result_axis!(fig[1,1], problems; kwargs...)
    
    # Save gap data if available
    if !isnothing(gap_result) && length(problems) == 1
        problem = problems[1]
        gap_value, phase_std, phase_grads, slope_std, slope_grad, ratio, metric = gap_result
        
        # Create gaps directory if it doesn't exist
        gaps_path = gaps_dir(problem)
        !isdir(gaps_path) && mkpath(gaps_path)
        
        # Save gaps data with proper symbol keys
        gaps_file = gaps_filepath(problem; metric=metric)
        jldsave(gaps_file;
            gap=gap_value,
            phase_standard=phase_std,
            phase_grads=phase_grads,
            slope_standard=slope_std,
            slope_grads=slope_grad,
            ratio=ratio,
            metric=string(metric)
        )
        @info "Saved gaps (metric=$metric) for $(get_name(problem)) to $gaps_file"
    end
    
    if save_plot
        save(plot_dir() * "/" * plot_name(problems) * ".png", fig)
        save(plot_dir() * "/" * plot_name(problems) * ".pdf", fig)
    end
    return fig
end

function plot_result_special!(figpos::GridPosition, fig::Figure, symbol::Symbol; kwargs...)
    (symbol == :nothing) && return

    if symbol == :legend
        # Find the first axis in the figure
        axis_with_data = nothing
        for obj in fig.content
            if obj isa Axis && !isempty(obj.scene.plots)
                axis_with_data = obj
                break
            end
        end
        
        if isnothing(axis_with_data)
            @warn "No axis with plots found for legend creation"
            return
        end
        
        Legend(figpos, axis_with_data, "Legend"; titleposition=:top)

    else
        @warn "Unknown special plot symbol: $symbol"
        @assert false
    end
end
function plot_result_special!(figpos::GridPosition, fig::Figure, str::String; kwargs...)
    Label(figpos, str)
end

function plot_result_axis!(figpos::GridPosition, problems::AbstractVector{<:AbstractProblem};
    xscale = log10,
    yscale = log10,
    legend = true,
    max_iters = typemax(Int),
    plot_individual_runs = false,
    metric = :tv,
    plotted_groups = nothing,
    compute_slope = true,
)
    @info "Plotting results for problems: $(get_name.(problems))"
    ################
    ### SETTINGS ###
    ################

    ### metric type based on symbol
    metric_type = if metric == :tv
        TVMetric
    elseif metric == :convergence
        TVMetric  # Use TVMetric as placeholder for file loading, but seek convergence files
    else
        error("Unknown metric symbol: $metric. Use :tv or :convergence.")
    end

    ### max plotted iters
    maxiter = nothing
    # maxiter = 100

    # TODO groups
    # plotted_groups = ["loglike-imiqr", "loglike", "standard", "eiv", "eiig", "nongp", "tnp"]
    # plotted_groups = ["standard", "est", "loglike", "loglike-imiqr"]
    # plotted_groups = ["standard", "loglike"]
    # plotted_groups = ["standard", "eiv", "eiig"]
    # plotted_groups = ["standard", "nongp", "bi", "tnp"]
    # plotted_groups = ["standard", "grads"]
    # plotted_groups = ["standard", "grads", "standard-lazy", "grads-lazy", "standard-lazy53x", "grads-lazy53x", "standard-lazy53x-singlerun", "grads-lazy53x-singlerun"]
    # plotted_groups = ["standard", "grads", "standard-lazy", "grads-lazy", "standard-warm", "grads-warm"]
    plotted_groups = isnothing(plotted_groups) ? ["standard-warm", "grads-warm"] : plotted_groups

    # a list of all groups is needed to keep plot colors consistent
    colors = Makie.wong_colors()
    main_groups = ["loglike", "standard", "eiv", "eiig", "immd", "nongp", "alt"]
    color_map = Dict(group => colors[i] for (i, group) in enumerate(main_groups))
    # Canonical paper palette override (2026-08-06): "standard" is yellow, not
    # colors[2]=orange (orange is reserved for niche custom-proxy plots
    # elsewhere). "maxvar" isn't in main_groups at all — without this it would
    # fall through to an arbitrary tab10 fallback color below, so Group B's
    # cross_all_tv_convergence.png (plotted_groups=["maxvar","eiv"]) wouldn't
    # match Group A's "standard"=yellow convention. Set both explicitly, before
    # the fallback loop, so the loop's `!haskey` check skips "maxvar".
    color_map["standard"] = colors[2]
    color_map["maxvar"]   = colors[2]

    # get some fallback colors for any additional groups
    extra_colors_iter = Iterators.Stateful(Iterators.cycle(Makie.colorschemes[:tab10]))
    for group in vcat("grads", plotted_groups) # TODO grads
        if !haskey(color_map, group)
            color_map[group] = popfirst!(extra_colors_iter)
        end
    end

    # include log versions of the problems as well
    add_log_variants!(problems)
    
    title = _get_plot_title(problems[1])

    # Generate ylabel based on metric symbol
    ylabel = if metric == :convergence
        "L2 risk (simulator)"
    else
        "TV"
    end
    ###

    # Load scores based on metric type - for convergence, manually load from convergence files
    if metric == :convergence
        scores_by_group = load_stored_convergence_scores(problems)
    else
        scores_by_group = load_stored_scores(problems, metric_type)
    end

    ax = Axis(figpos; xlabel="simulations",
        ylabel,
        title,
        xscale,
        yscale,
        # ygridvisible = false,
        # yminorticks,
        # yminorticksvisible = true,
        # yminorgridvisible = true,
        # yminorgridcolor = RGBAf(0, 0, 0, 0.12),
    )

    # Super hacky way to order the data same as in `plotted_groups`
    function prepare_data(problem_group, scores)
        parts = split(problem_group, "_")
        group = parts[end]
        pname = join(parts[1:end-1], "_")
        return pname, group, scores
    end
    plot_data_ = [prepare_data(problem_group, scores) for (problem_group, scores) in scores_by_group]
    plot_data = similar(plot_data_, 0)
    for group in plotted_groups
        for t in plot_data_
            # TODO only plotted_groups OR any group which contains the group as substring
            # (t[2] == group) && push!(plot_data, t)
            occursin(group, t[2]) && push!(plot_data, t)
        end
    end

    # Track whether we've added the differentiation note
    grads_noted = false
    
    # Track slopes and linear phase ranges for each group (base groups only)
    slopes_by_base_group = Dict{String, Float64}()
    phase_ranges_by_base_group = Dict{String, Tuple{Float64, Float64, Float64}}()  # (linear_start_x, linear_end_x, region_strength)
    median_scores_by_base_group = Dict{String, Tuple{Vector, Vector}}()  # (xs, median_scores) for each group
    
    # Initialize gap_result to store computed gap
    gap_result = nothing
    
    # plotting
    used_labels = Set{String}()
    for (pname, group, scores) in plot_data
        label = get_run_label(group)
        if label in used_labels
            label = nothing  # avoid duplicate legend entries
        else
            push!(used_labels, label)
        end
        color = get(color_map, group, nothing)
        style = :solid
        
        ### proxy variants
        # TODO
        # if pname in ["AbsABProblem", "DiffusionProblem2", "ProxySIRProblem"]
        #     @assert group == "standard"
        #     color = color_map["alt"]
        #     # TODO
        #     label = "GP - alt. proxy - MaxVar"
        #     # label = "GP - bad proxy - MaxVar"
        #     # label = "GP - good proxy - MaxVar"
        # end
        ### special styles
        if group == "est"
            color = color_map["standard"]
            style = :dash
        end
        if group == "loglike-imiqr"
            color = color_map["loglike"]
            style = :dash
        end
        if startswith(group, "standard")
            color = color_map["standard"]
        end
        if startswith(group, "grads")
            color = color_map["grads"]
        end
        if startswith(group, "uniform")
            color = occursin("grads", group) ? color_map["grads"] : color_map["standard"]
        end
        if startswith(group, "maxvar")
            color = occursin("grads", group) ? color_map["grads"] : color_map["maxvar"]
        end
        ### fallback colors
        @assert !isnothing(color)

        # TODO rem
        # if ((pname == "SIRProblem") || (pname == "ProxySIRProblem")) && (group == "grads")
        #     @warn "Truncating \"grads\" runs to 50 iterations only!"
        #     scores = [s[1:50] for s in scores]
        # end

        # scores is a Vector of score histories (each is a Vector)
        # Pad with `missing` to equal length if needed
        maxlen = maximum(length.(scores))
        maxlen = min(maxlen, max_iters)
        
        if length(scores) != 20
            @warn "Group \"$group\" has only $(length(scores)) runs, expected 20."
        end
        if any([any(isnan.(s)) for s in scores])
            have_nans = [any(isnan.(s)) for s in scores]
            @warn "Group \"$group\" has NaN values in runs: $(findall(have_nans))."
            @warn "Excluding runs with NaNs from the plot."
            scores = scores[.!have_nans]
            isempty(scores) && continue
        end
        if !allequal(length.(scores))
            max_run_len = maximum(length.(scores))
            successful = sum(length.(scores) .== max_run_len)
            failed = findall(length.(scores) .!= max_run_len)
            @warn "Scores for group \"$group\" have different lengths!
            ($successful/$(length(scores)) runs have the max length of $max_run_len,
            runs $failed have shorter lengths.)"

            ### pad or align
            # @warn "Padding with `missing`."
            # padded = pad_with_missing.(scores, Ref(maxlen))
            # arr = reduce(hcat, padded)
            @warn "Discarding run ends to include only iterations with full data."
            aligned = align_scores(scores)
            arr = reduce(hcat, aligned)
        else
            arr = reduce(hcat, scores)
        end


        ### init data points
        init_data = 3
        if endswith(group, "-lazy")
            init_data = 50
        end
        xs = init_data:init_data+maxlen-1
        
        # Plot individual runs if requested
        if plot_individual_runs
            for i in eachindex(scores)
                run_scores = scores[i] # instead of arr[:, i] to get the original untruncated data
                run_scores_trimmed = isnothing(maxiter) ? run_scores : run_scores[1:min(length(run_scores), maxiter)]
                lines!(ax, xs[eachindex(run_scores_trimmed)], run_scores_trimmed; color=color, alpha=0.8, linewidth=0.5)
            end
        end
        
        # Plot median line
        median_scores = mapslices(median∘skipmissing, arr; dims=2)[:]
        isnothing(maxiter) || (median_scores = median_scores[1:min(length(median_scores), maxiter)])
        lines!(ax, xs[eachindex(median_scores)], median_scores; label, color=color, linestyle=style, linewidth=2)
        
        # Compute convergence slope and phase boundaries
        if compute_slope
            try
                slope, init_end_idx, learning_end_idx, init_x, learning_end_x, region_strength = compute_convergence_slope(xs[eachindex(median_scores)], median_scores)
                @info "Computed slope for $group: linear_start=$init_x, linear_end=$learning_end_x (indices $init_end_idx to $learning_end_idx, region_strength=$region_strength)"
                # Store slope by base group (without modifiers like -warm-noise)
                # Preserve compound names like uniform-grads; strip only trailing modifiers like -warm, -noise=...
                base_group = if startswith(group, "uniform-grads")
                    "uniform-grads"
                elseif startswith(group, "uniform")
                    "uniform"
                elseif startswith(group, "grads")
                    "grads"
                else
                    split(group, "-")[1]
                end
                slopes_by_base_group[base_group] = slope
                # Store phase boundaries (start and end of linear region) plus region strength
                phase_ranges_by_base_group[base_group] = (init_x, learning_end_x, region_strength)
                # Store median scores and xs values for later gap computation
                median_scores_by_base_group[base_group] = (collect(xs[eachindex(median_scores)]), median_scores)
                @info "Stored phase range for $base_group"
            catch e
                @warn "Failed to compute slope for group \"$group\": $e"
            end
        end
        
        # noise level note
        if contains(group, "-noise=")
            noise_val = split(group, "-noise=")[2]
            x_pos = xs[eachindex(median_scores)[end]]
            y_pos = median_scores[end]
            text!(ax, x_pos, y_pos; text="noise=$noise_val", align=(:right, :bottom), fontsize=10, color=color)
        end

        if contains(group, "grads")
            problem = reconstruct_problem(pname)
            x_dim_ = x_dim(problem)
            y_dim_ = y_dim(problem)
            
            # Add differentiation note once per axis
            if !grads_noted
                is_forward = x_dim_ <= y_dim_
                diff_type = is_forward ? "Forward diff." : "Reverse diff."
                diff_color = is_forward ? :darkgreen : :darkred
                text!(ax, 0.05, 0.08, text=diff_type; align=(:left, :bottom), space=:relative, fontsize=14, color=diff_color)
                grads_noted = true
            end
            
            # TODO uncomment:

            # # Cost-adjusted: multiply by (1 + y_dim) for adjoint gradients, or (1 + x_dim) for forward gradients
            # cost_multiplier = y_dim_ < x_dim_ ? (1 + y_dim_) : (1 + x_dim_)
            # xs_cost_adj = cost_multiplier .* xs
            # label_cost_adj = label * " - cost-adjusted"
            # lines!(ax, xs_cost_adj[eachindex(median_scores)], median_scores; label=label_cost_adj, color=color, linestyle=:dash, linewidth=2)
            
            # # Data-adjusted: total scalar data = y_dim outputs + x_dim*y_dim gradients = y_dim*(1 + x_dim)
            # xs_data_adj = (1 + x_dim_) .* xs
            # label_data_adj = label * " - data-adjusted"
            # lines!(ax, xs_data_adj[eachindex(median_scores)], median_scores; label=label_data_adj, color=color, linestyle=:dot, linewidth=2)
        end

        # # Plot quantile band with alpha
        # lq = mapslices(x -> quantile(skipmissing(x), 0.1), arr; dims=2)[:]
        # uq = mapslices(x -> quantile(skipmissing(x), 0.9), arr; dims=2)[:]
        # band!(ax, xs, lq, uq; color=color, alpha=0.6)

        # # Plot min/max dotted lines
        # maxs = mapslices(x -> maximum(skipmissing(x)), arr; dims=2)[:]
        # mins = mapslices(x -> minimum(skipmissing(x)), arr; dims=2)[:]
        # lines!(ax, xs, maxs; color=color, linestyle=:dot, linewidth=1)
        # lines!(ax, xs, mins; color=color, linestyle=:dot, linewidth=1)
    end

    # Draw vertical lines showing the phase boundaries
    @info "Phase ranges: $phase_ranges_by_base_group"
    for (base_group, (linear_start_x, linear_end_x, region_strength)) in phase_ranges_by_base_group
        if haskey(slopes_by_base_group, base_group)
            color = if startswith(base_group, "uniform")
                occursin("grads", base_group) ? color_map["grads"] : color_map["standard"]
            else
                get(color_map, base_group, :gray)
            end
            @info "Drawing lines for group $base_group: start=$linear_start_x, end=$linear_end_x"
            # Draw line at start of linear phase
            vlines!(ax, [linear_start_x]; color=color, linestyle=:dot, linewidth=3, alpha=0.5)
            # Draw line at end of linear phase
            vlines!(ax, [linear_end_x]; color=color, linestyle=:dot, linewidth=3, alpha=0.3)
        end
    end

    # Display convergence statistics in upper right corner
    # Find a no-grads key and a grads key for comparison
    group_keys = collect(keys(slopes_by_base_group))
    std_key  = findfirst(k -> !occursin("grads", k), group_keys)
    grad_key = findfirst(k ->  occursin("grads", k), group_keys)

    group_color(g) = if startswith(g, "uniform")
        occursin("grads", g) ? color_map["grads"] : color_map["standard"]
    else
        get(color_map, occursin("grads", g) ? "grads" : "standard", :gray)
    end

    if length(slopes_by_base_group) >= 2 && !isnothing(std_key) && !isnothing(grad_key)
        sk = group_keys[std_key]
        gk = group_keys[grad_key]
        slope_std = slopes_by_base_group[sk]
        slope_grad = slopes_by_base_group[gk]

        ratio = slope_grad / slope_std

        gap = nothing
        if haskey(median_scores_by_base_group, sk) && haskey(median_scores_by_base_group, gk) &&
           haskey(phase_ranges_by_base_group, sk) && haskey(phase_ranges_by_base_group, gk)
            xs_std, ys_std = median_scores_by_base_group[sk]
            xs_grads, ys_grads = median_scores_by_base_group[gk]
            phase_std = phase_ranges_by_base_group[sk]
            phase_grads = phase_ranges_by_base_group[gk]

            gap = compute_constant_gap(xs_std, ys_std, ys_grads, phase_std, phase_grads)
            gap_result = (gap, phase_std, phase_grads, slope_std, slope_grad, ratio, metric)
        end

        text!(ax, 0.98, 0.98; text=@sprintf("slope: %.2f", slope_std),
              align=(:right, :top), space=:relative, fontsize=11, color=group_color(sk))
        text!(ax, 0.98, 0.88; text=@sprintf("slope: %.2f", slope_grad),
              align=(:right, :top), space=:relative, fontsize=11, color=group_color(gk))
        ratio_str = isnothing(gap) ? @sprintf("ratio: %.2f", ratio) :
                                     @sprintf("ratio: %.2f  gap: %.2e", ratio, gap)
        text!(ax, 0.98, 0.78; text=ratio_str,
              align=(:right, :top), space=:relative, fontsize=11, color=:black)
    elseif length(slopes_by_base_group) >= 1
        gk = first(keys(slopes_by_base_group))
        slope = slopes_by_base_group[gk]
        text!(ax, 0.98, 0.98; text=@sprintf("slope: %.2f", slope),
              align=(:right, :top), space=:relative, fontsize=11, color=group_color(gk))
    end
    # plot reference opt_mmd values
    if metric == OptMMDMetric
        p = problems[1]
        ref_sample_count = 200
        @warn "CHECK THAT THE OptMMD METRIC HAS BEEN CALCULATED WITH $ref_sample_count SAMPLES!"
        ref_optmmd_file = joinpath(data_dir(p), "opt_mmd", "mmd_vals_$(ref_sample_count).jld2")
        if isfile(ref_optmmd_file)
            mmd_vals = load(ref_optmmd_file)["mmd_vals"]
            lq = quantile(mmd_vals, 0.1)
            med = median(mmd_vals)
            uq = quantile(mmd_vals, 0.9)

            hlines!(ax, [lq, uq]; color=:black, linestyle=:dot)
            hlines!(ax, [med]; color=:black, linestyle=:dash)
        else
            @warn "Reference OptMMD file not found: $ref_optmmd_file"
        end
    end

    if legend
        try
            axislegend(ax; position=:lb)
        catch
        end
    end

    return ax, gap_result
end

function _get_plot_title(problem::SharpProblem)
    return _get_plot_title(problem.base) * " (sharp)"
end

function _get_plot_title(problem::HexObsProblem)
    return _get_plot_title(problem.inner) * " (hex)"
end

function _get_plot_title(problem::CrossPolytopeObsProblem)
    return _get_plot_title(problem.inner) * " (cross)"
end

function _get_plot_title(problem::AbstractProblem)
    title = problem |> typeof |> nameof |> string

    # TODO rem
    if startswith(title, "MeanGauss")
        return title * " $(problem.x_dim)D"
    end

    title = title[1:end-7]  # remove "Problem" suffix

    # TODO rem
    if title == "ProxySIR"
        @warn "Renaming title for the ProxySIRProblem."
        title = "SIR (with proxy)"
    elseif title == "SIR"
        @warn "Renaming title for the SIRProblem."
        title = "SIR (without proxy)"
    elseif title == "BealeProxy"
        title = "Beale (proxy)"
    elseif title == "GoldsteinPriceProxy"
        title = "Goldstein-Price (proxy)"
    elseif title == "Multidim"
        @warn "Renaming title for a Multidim problem."
        base_problem_ = problem.problem
        total_dim_ = problem.scaleup * x_dim(base_problem_)
        title = base_problem_ |> typeof |> nameof |> string
        title = title[1:end-7]  # remove "Problem" suffix
        title *= " $(total_dim_)D"
        @show title
    end
    return title
end

plot_name(problems::AbstractVector) = join(plot_name.(problems), "_")
plot_name(problem::AbstractProblem) = string(typeof(problem))

function add_log_variants!(problems::AbstractVector{<:AbstractProblem})
    for p in problems
        pname = p |> typeof |> nameof |> string
        startswith(pname, "Log") && continue
        logpname = "Log" * pname
        if hasproperty(Main, Symbol(logpname))
            logp = getproperty(Main, Symbol(logpname))()
            @warn "Adding $logpname to the list of plotted problems."
            (logp in problems) || push!(problems, logp)
        end
    end
    return problems
end

function load_stored_scores(problems::AbstractVector, metricT::Type{<:DistributionMetric}; kwargs...)
    dicts = load_stored_scores.(problems, Ref(metricT); kwargs...)

    # Find all keys and check for duplicates
    all_keys = reduce(vcat, [collect(keys(d)) for d in dicts])
    key_counts = Dict(k => count(==(k), all_keys) for k in unique(all_keys))
    common_keys = [k for (k, v) in key_counts if v > 1]
    @assert isempty(common_keys)

    return merge(dicts...)
end
function load_stored_scores(problem::AbstractProblem, metricT::Type{<:DistributionMetric}; expected_runs=20)
    ### dir
    dir = data_dir(problem)
    # dir = "data/archive/data_01/" * string(typeof(problems[1]))
    files = sort(Glob.glob(joinpath(dir, "*.jld2")))
    
    # TODO indices
    # get files ordered by indices -- useful later
    split_fname(f) = split(rsplit(basename(f), "."; limit=2)[1], r"[_=]") # TODO remove =, put back just "_"
    files = filter(f -> !isnothing(tryparse(Int, split_fname(f)[2])), files)
    indices_ = [parse(Int, split_fname(f)[2]) for f in files]
    perm_ = sortperm(indices_)
    files = files[perm_]

    scores_by_group = Dict{String, Vector{Vector{Float64}}}()
    indices_by_group = Dict{String, Vector{Int}}() # TODO indices

    for file in files
        # fname, suffix = rsplit(basename(file), "."; limit=2) # only split by the last dot
        fname_parts = split_fname(basename(file))
        fname = join(fname_parts)
        suffix = fname_parts[end]
        
        # (split(fname, "_")[1] == "test") && continue  # skip test files
        # group = split(fname, "_")[1]

        # TODO remove: special handling for the noise values written in the file names
        if endswith(fname_parts[1], "-noise")
            group = get_name(problem) * "_" * fname_parts[1] * "=" * fname_parts[3]
        else
            group = get_name(problem) * "_" * fname_parts[1]
        end
        
        endswith(fname, metric_fname(metricT)) || continue  # only consider the metric files

        if !haskey(scores_by_group, group)
            scores_by_group[group] = Vector{Vector{Float64}}()
            indices_by_group[group] = Vector{Int}() # TODO indices
        end
        push!(scores_by_group[group], load(file, "score"))
        push!(indices_by_group[group], parse(Int, fname_parts[2])) # TODO indices
    end

    # TODO indices
    for (group, indices) in indices_by_group
        if length(indices) != expected_runs
            missing_runs = setdiff(1:expected_runs, indices)
            @warn "Group \"$group\" on problem $(get_name(problem)) has only $(length(indices)) runs, expected $expected_runs.
            missing runs: $missing_runs"
        end
    end

    return scores_by_group
end

function load_stored_convergence_scores(problems::AbstractVector)
    dicts = load_stored_convergence_scores.(problems)

    # Find all keys and check for duplicates
    all_keys = reduce(vcat, [collect(keys(d)) for d in dicts])
    key_counts = Dict(k => count(==(k), all_keys) for k in unique(all_keys))
    common_keys = [k for (k, v) in key_counts if v > 1]
    @assert isempty(common_keys)

    return merge(dicts...)
end

function load_stored_convergence_scores(problem::AbstractProblem; expected_runs=20)
    ### dir
    dir = data_dir(problem)
    files = sort(Glob.glob(joinpath(dir, "*.jld2")))
    
    # get files ordered by indices
    split_fname(f) = split(rsplit(basename(f), "."; limit=2)[1], r"[_=]")
    files = filter(f -> !isnothing(tryparse(Int, split_fname(f)[2])), files)
    indices_ = [parse(Int, split_fname(f)[2]) for f in files]
    perm_ = sortperm(indices_)
    files = files[perm_]

    scores_by_group = Dict{String, Vector{Vector{Float64}}}()
    indices_by_group = Dict{String, Vector{Int}}()

    for file in files
        fname_parts = split_fname(basename(file))
        fname = join(fname_parts)
        suffix = fname_parts[end]
        
        # Check if this is a convergence file
        if endswith(fname_parts[1], "-noise")
            group = get_name(problem) * "_" * fname_parts[1] * "=" * fname_parts[3]
        else
            group = get_name(problem) * "_" * fname_parts[1]
        end
        
        endswith(fname, "convergence") || continue  # only consider convergence files

        if !haskey(scores_by_group, group)
            scores_by_group[group] = Vector{Vector{Float64}}()
            indices_by_group[group] = Vector{Int}()
        end
        push!(scores_by_group[group], vec(mean(load(file, "score"), dims=1)))  # Average across output dims
        push!(indices_by_group[group], parse(Int, fname_parts[2]))
    end

    for (group, indices) in indices_by_group
        if length(indices) != expected_runs
            missing_runs = setdiff(1:expected_runs, indices)
            @warn "Group \"$group\" on problem $(get_name(problem)) has only $(length(indices)) runs, expected $expected_runs.
            missing runs: $missing_runs"
        end
    end

    return scores_by_group
end

function pad_with_missing(v::AbstractVector, len::Int)
    if length(v) < len
        return vcat(v, fill(missing, len - length(v)))
    else
        return v[1:len]
    end
end

function align_scores(scores::AbstractVector{<:AbstractVector})
    lens = length.(scores)
    minlen = minimum(lens)
    @show minlen
    return [s[1:minlen] for s in scores]
end

function common_prefix(strs::AbstractVector{String})
    isempty(strs) && return ""
    minlen = minimum(length.(strs))
    prefix = strs[1][1:minlen]
    for s in strs[2:end]
        for i in eachindex(prefix)
            if s[i] != prefix[i]
                prefix = prefix[1:i-1]
                break
            end
        end
    end
    return prefix
end

"""
    plot_gaps_vs_dimension(; save_plot=false, base_fontsize=20, r_c=2.0, ν=2.5, metric=:tv, legend=true)

Plot the stored gap values (constant vertical offset between standard and gradient-augmented approaches)
as a function of problem dimension. Includes both empirical gaps and theoretical predictions.

The gap represents the constant factor by which the standard approach is worse than the gradient-augmented 
approach in terms of error reduction per unit cost, manifesting as a vertical offset on log-log plots.

Args:
    save_plot: If true, save the figure to plots/gaps_vs_dimension_{metric}.png
    base_fontsize: Base font size for the plot
    r_c: Relative cost of gradient evaluation (default 2.0 for reverse diff with y_dim=1)
    ν: Matérn smoothness parameter (default 2.5)
    metric: Which metric to plot (:tv for TV metric on posterior, :convergence for L2 risk on simulator)
    legend: If true, show the legend (default true)

Returns:
    Figure with gap values and theoretical predictions
"""
function plot_gaps_vs_dimension(; save_plot=false, base_fontsize=20, r_c=2.0, ν=2.5, metric=:tv, legend=true)
    set_theme_fonts!(base_fontsize=base_fontsize)
    
    # Collect gap data from all problems
    gaps_data = Dict{String, Tuple{Float64, Int}}()  # problem_name => (gap_value, dimension)
    
    # Scan all problem directories for gap files
    data_base = "data-convergence4"
    if !isdir(data_base)
        @warn "Data directory not found: $data_base"
        return nothing
    end
    
    problem_dirs = filter(isdir, [joinpath(data_base, d) for d in readdir(data_base)])
    
    for pdir in problem_dirs
        gaps_dir_path = joinpath(pdir, "gaps")
        gaps_file = joinpath(gaps_dir_path, "gaps_$(metric).jld2")
        
        if isfile(gaps_file)
            try
                gap_data = load(gaps_file)
                gap_value = gap_data["gap"]
                
                # Extract dimension from directory name or reconstruct problem
                pname = basename(pdir)
                
                # Try to determine dimension
                dimension = nothing
                
                # Try to parse dimension directly from name (e.g., "MeanGauss3", "MultidimProblem1", etc.)
                for match in eachmatch(r"(\d+)$", pname)
                    dimension = parse(Int, match[1])
                    break
                end
                
                # If not found, try to reconstruct the problem
                if isnothing(dimension)
                    try
                        if haskey(Main, Symbol(pname))
                            problem = getproperty(Main, Symbol(pname))()
                            if hasproperty(problem, :x_dim)
                                dimension = problem.x_dim
                            elseif hasproperty(problem, :scaleup)
                                base_problem = problem.problem
                                base_x_dim = x_dim(base_problem)
                                dimension = problem.scaleup * base_x_dim
                            else
                                # Try using x_dim function
                                dimension = x_dim(problem)
                            end
                        end
                    catch e
                        @warn "Could not determine dimension for problem $pname: $e"
                    end
                end
                
                if !isnothing(dimension)
                    gaps_data[pname] = (gap_value, dimension)
                end
            catch e
                @warn "Failed to load gaps from $gaps_file: $e"
            end
        end
    end
    
    if isempty(gaps_data)
        @warn "No gap data found in $data_base"
        return nothing
    end
    
    # Sort by dimension
    problems_sorted = sort(collect(gaps_data); by = kv -> kv[2][2])
    dims = [kv[2][2] for kv in problems_sorted]
    gap_vals = [kv[2][1] for kv in problems_sorted]  # Raw values (not log10)
    labels_vec = [kv[1] for kv in problems_sorted]
    
    # Create figure
    fig = Figure()
    
    # Construct title based on metric
    metric_label = if metric == :convergence
        "L2 risk (simulator)"
    else
        "TV metric (posterior)"
    end
    
    ax = Axis(fig[1, 1];
        xlabel = "Problem Dimension (d)",
        ylabel = "Gap",
        yscale = log10,
        title = "Constant Gap between Standard and Gradient-Augmented Approaches\n($metric_label)"
    )
    
    # Plot empirical gaps
    scatter!(ax, dims, gap_vals; label="Empirical gaps", markersize=10, color=(:blue, 0.7))
    
    # Add labels for each point
    for (i, (dim, gap, label)) in enumerate(zip(dims, gap_vals, labels_vec))
        text!(ax, dim, gap; text=label, align=(:left, :center), fontsize=base_fontsize-4, offset=(5, 0))
    end
    
    # Compute and plot theoretical predictions
    if !isempty(dims)
        # Theory: gap ≈ (1+d) / r_c (in actual values, displayed in log scale by axis)
        d_range = 1:maximum(dims)
        theoretical_gaps = (1 .+ d_range) ./ r_c
        
        lines!(ax, d_range, theoretical_gaps; label="Theory: (1+d)/r_c", 
               linestyle=:dash, linewidth=2, color=(:red, 0.7))
    end
    
    if legend
        axislegend(ax; position=:lb)
    end
    
    if save_plot
        !isdir(plot_dir()) && mkpath(plot_dir())
        save(plot_dir() * "/gaps_vs_dimension_$(metric).png", fig)
        @info "Saved gap plot to $(plot_dir())/gaps_vs_dimension_$(metric).png"
    end
    
    return fig
end

@kwdef struct CustomLogMinorTicks
    logstep::Real = 1
end

function Makie.get_minor_tickvalues(ticks::CustomLogMinorTicks, scale, tickvalues, vmin, vmax)
    return get_minor_ticks(ticks.logstep, vmin, vmax)
end
function get_minor_ticks(logstep, vmin, vmax)
    start = log(vmin) ÷ logstep * logstep
    stop = log(vmax) ÷ logstep * logstep
    ticks = [exp(t) for t in start:logstep:stop]
    # labels = [L"e^{%$t}" for t in start:logstep:stop]
    return ticks
end
