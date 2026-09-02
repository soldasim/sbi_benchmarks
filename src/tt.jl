###
### This file contains miscellaneous scripts for plotting and testing.
###

using BOSIP
using CairoMakie
using CairoMakie.Colors

include("main.jl")

function set_theme_fonts!(; base_fontsize=20, increase_labels=0)
    set_theme!(
        fontsize = base_fontsize,
        Axis = (
            titlesize = base_fontsize + 2,
            xlabelsize = base_fontsize + increase_labels,
            ylabelsize = base_fontsize + increase_labels,
            xticklabelsize = base_fontsize - 2,
            yticklabelsize = base_fontsize - 2,
        ),
        Legend = (
            titlesize = base_fontsize,
            labelsize = base_fontsize - 1 + increase_labels,
        ),
        Label = (
            textsize = base_fontsize + increase_labels,
        ),
        Colormap = (
            label = base_fontsize + increase_labels,
        ),
    )
end

function plot_sir_comparison(ydim::Int; display=false)
    @assert 1 <= ydim <= 10
    
    # Create both problem types
    sir_problem = SIRProblem()
    proxy_sir_problem = ProxySIRProblem()
    
    # Get problem properties (assuming both have same parameter bounds)
    bounds = domain(sir_problem).bounds
    sir_sim = simulator(sir_problem)
    proxy_sim = simulator(proxy_sir_problem)
    x_len = length(bounds[1])
    
    # Ensure we have a 2D parameter space
    @assert x_len == 2 "This function is designed for 2D parameter spaces"
    
    # Create parameter grids
    resolution = 200
    x1_range = range(bounds[1][1], bounds[2][1], length=resolution)
    x2_range = range(bounds[1][2], bounds[2][2], length=resolution)
    
    # Evaluate both simulators over the parameter grid
    sir_values = zeros(resolution, resolution)
    proxy_values = zeros(resolution, resolution)
    
    for (i, x1) in enumerate(x1_range)
        for (j, x2) in enumerate(x2_range)
            x = [x1, x2]
            sir_y = sir_sim(x)
            proxy_y = proxy_sim(x)
            sir_values[j, i] = sir_y[ydim]
            proxy_values[j, i] = proxy_y[ydim]
        end
    end
    
    # Create the comparison figure with space for individual colorbars
    fig = Figure(size=(1400, 500))
    
    # SIR Problem plot
    ax1 = Axis(fig[1, 1],
        xlabel="Parameter 1",
        ylabel="Parameter 2", 
        title="SIR Output Dimension $ydim",
        aspect=AxisAspect(1)
    )
    
    hm1 = heatmap!(ax1, x1_range, x2_range, sir_values, 
        colormap=:tempo)
    # Add single contour at prior mean for this output dimension
    try
        level1 = prior_mean(sir_problem)[ydim]
        contour!(ax1, x1_range, x2_range, sir_values, levels=[level1], color=:red, linewidth=2)
    catch e
        @warn "Could not add prior-mean contour to SIR plot: $e"
    end

    # Add colorbar for SIR plot
    Colorbar(fig[1, 2], hm1, label="SIR Output")
    
    # ProxySIR Problem plot
    ax2 = Axis(fig[1, 3],
        xlabel="Parameter 1",
        ylabel="Parameter 2", 
        title="ProxySIR Output Dimension $ydim",
        aspect=AxisAspect(1)
    )
    
    hm2 = heatmap!(ax2, x1_range, x2_range, proxy_values, 
        colormap=:tempo)
    # Add single contour at prior mean for this output dimension (proxy)
    try
        level2 = prior_mean(proxy_sir_problem)[ydim]
        contour!(ax2, x1_range, x2_range, proxy_values, levels=[level2], color=:red, linewidth=2)
    catch e
        @warn "Could not add prior-mean contour to ProxySIR plot: $e"
    end

    # Add colorbar for ProxySIR plot
    Colorbar(fig[1, 4], hm2, label="ProxySIR Output")
    
    # Add overall title
    Label(fig[0, :], "SIR vs ProxySIR Comparison - Output Dimension $ydim", 
          fontsize=16, font="bold")
    
    display && CairoMakie.display(fig)
    return fig
end

function plot_sir_output(ydim::Int; use_proxy::Bool=true, base_fontsize::Int=20, save_plot=false)
    @assert 1 <= ydim <= 10
    set_theme_fonts!(; base_fontsize, increase_labels=2)

    # Create SIR problem based on the flag
    problem = if use_proxy
        ProxySIRProblem()
    else
        SIRProblem()
    end
    
    # Get problem properties
    bounds = domain(problem).bounds
    sim = simulator(problem)
    x_len = length(bounds[1])
    
    # Ensure we have a 2D parameter space
    @assert x_len == 2 "This function is designed for 2D parameter spaces"
    
    # Create parameter grids
    resolution = 200
    x1_range = range(bounds[1][1], bounds[2][1], length=resolution)
    x2_range = range(bounds[1][2], bounds[2][2], length=resolution)
    
    # Evaluate simulator over the parameter grid
    output_values = zeros(resolution, resolution)
    
    for (i, x1) in enumerate(x1_range)
        for (j, x2) in enumerate(x2_range)
            x = [x1, x2]
            y = sim(x)
            output_values[j, i] = y[ydim]  # Note: j,i for proper orientation
        end
    end
    
    # Create the plot
    problem_name = use_proxy ? "ProxySIR" : "SIR"
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xlabel = L"x_1 \:\coloneq\: \beta",
        ylabel = L"x_2 \:\coloneq\: \gamma", 
        title = use_proxy ? "custom proxy variable" : "simulator output",
        aspect=AxisAspect(1)
    )
    
    # Create heatmap
    hm = heatmap!(ax, x1_range, x2_range, output_values, 
        colormap=:tempo)
    
    # Add single contour at prior mean for this output dimension (proxy)
    try
        problem = use_proxy ? ProxySIRProblem() : SIRProblem()
        level = prior_mean(problem)[ydim]
        # draw a single highlighted contour at the prior-mean level and add a legend entry
        cont = contour!(ax, x1_range, x2_range, output_values, levels=[level], color=:red, linewidth=2)
        # show legend for the contour (position: right-top)
        label = use_proxy ? L"\arg\max_x\: p(z_{o,%$(ydim)}|x)" : L"\arg\max_x\: p(z_{o,%$(ydim)}|x)"
        lines!(ax, Float64[], Float64[], label = label, color = :red)
        axislegend(ax; position = :rt)
    catch e
        @warn "Could not add prior-mean contour to plot: $e"
    end

    # Add colorbar
    label_str = use_proxy ? L"\delta_%$(ydim) \:\coloneq\: \log(1 + I(t_%$(ydim)))" : L"y_%$(ydim) \:\coloneq\: I(t_%$(ydim))"
    Colorbar(fig[1, 2], hm, label = label_str)
    
    # # Add contour lines for better visualization
    # contour!(ax, x1_range, x2_range, output_values, 
    #     levels=10, color=:white, linewidth=1, alpha=0.5)
    
    fname = use_proxy ? "plots/sir_response_proxy.pdf" : "plots/sir_response_output.pdf"
    save_plot && save(fname, fig)
    return fig
end

function plot_outputs(problem::AbstractProblem)
    bounds = domain(problem).bounds
    sim = simulator(problem)
    y_len = y_dim(problem)
    x_len = length(bounds[1])
    
    # xt = x_transform(problem)
    xt = nothing

    # TODO
    # x0 = [1.0, 1.0]
    x0 = SIRModule.x_ref
    # x0 = DuffingModule.x_ref
    # x0 = DiffusionModule.x_ref

    # TODO
    # std_obs = nothing
    # std_obs = DuffingModule.std_obs
    # std_obs = DiffusionModule.std_obs
    std_obs = SIRModule.std_obs

    figs = Figure(size = (400 * x_len, 250 * y_len))

    for x_dim in 1:x_len
        lb, ub = bounds[1][x_dim], bounds[2][x_dim]
        xs = range(lb, ub; length=200)
        
        if isnothing(xt)
            ### V1: without x transformation
            x_ = deepcopy(x0)
            ys = [sim(setindex!(x_, x, x_dim)) for x in xs]
            ys_mat = hcat(ys...)  # each column is output for one x

        else
            xt0 = xt(x0)
            xs_ = collect(xs)

            ### V2: with x transformation
            x_ = deepcopy(x0)
            ys_mat = zeros(y_len, length(xs))
            for (idx, xi) in enumerate(xs)
                x_[x_dim] = xi
                xt_ = xt(x_)
                @assert (xt_[begin:x_dim-1] == xt0[begin:x_dim-1]) && (xt_[x_dim+1:end] == xt0[x_dim+1:end]) "The plotting code with `x_transform` assumes that each x dimension is transformed independently."
                xs_[idx] = xt_[x_dim]
                ys_mat[:,idx] .= sim(xt_)
            end

            # plot the transformed values
            xs = xs_
            # ys_mat is already calculated with the transformed xs
        end

        # TODO
        y0 = prior_mean(problem)

        for i in 1:y_len
            ax = Axis(figs[i, x_dim], xlabel="Input dim $x_dim", ylabel="Output dim $i")
            lines!(ax, xs, ys_mat[i, :])
            if !isnothing(y0)
                if !isnothing(std_obs)
                    band!(ax, xs, 
                        fill(y0[i] - std_obs[i], length(xs)), 
                        fill(y0[i] + std_obs[i], length(xs)); 
                        color=(:red, 0.2)
                    )
                end
                hlines!(ax, y0[i]; color=:red, linestyle=:dash, label="GP Prior Mean")
            end
            if !isnothing(x0)
                vlines!(ax, x0[x_dim]; color=:red, linestyle=:dot, label="x₀")
            end
        end
    end
    return figs
end

function plot_duffing()
    x_ref = DuffingModule.x_ref
    sol = DuffingModule.duffing_simulation(x_ref)

    pos = [sol.u[i][1] for i in 1:length(sol.t)]
    positions, velocities, times = DuffingModule.extract_measurements(sol)
    
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="Position")
    lines!(ax, sol.t, pos, label="Position")
    scatter!(ax, times, positions; color=:red, label="Observations")
    axislegend(ax)
    return fig
end

function test_sol(problem::AbstractProblem, p::BosipProblem)
    ref = reference(problem)
    @assert ref isa Function
    true_post = x -> exp(ref(x))

    # TODO
    # x_ref = nothing
    x_ref = DuffingModule.x_ref

    true_ps = true_post.(eachcol(p.problem.data.X))
    best_i = argmax(true_ps)

    x_best = p.problem.data.X[:, best_i]
    p_best = true_ps[best_i]
    p_ref = isnothing(x_ref) ? nothing : true_post(x_ref)
    
    @show x_best
    @show p_best
    @show x_ref
    @show p_ref

    return nothing
end

function find_closest(xs::AbstractMatrix)
    count = size(xs, 2)
    idx = nothing
    best = Inf

    for i in 1:count
        for j in (i+1):count
            d = norm(xs[:, i] - xs[:, j])
            if d < best
                best = d
                idx = (i, j)
            end
        end
    end

    return xs[:,idx[1]], xs[:,idx[2]], idx
end
function find_closest(xs::AbstractMatrix, x::AbstractVector)
    count = size(xs, 2)
    idx = nothing
    best = Inf

    for i in 1:count
        d = norm(xs[:, i] - x)
        if d < best
            best = d
            idx = i
        end
    end

    return xs[:,idx], idx
end

function without(xs::AbstractMatrix, idx::AbstractVector{<:Integer})
    mask = trues(size(xs, 2))
    mask[idx] .= false
    return xs[:, mask]
end

function find_first_same(xs::AbstractMatrix; atol=1e-2)
    count = size(xs, 2)

    for i in 1:count
        for j in 1:(i-1)
            if isapprox(xs[:,j], xs[:,i]; atol)
                return xs[:,j], xs[:,i], (j, i)
            end
        end
    end

    return nothing
end

function plot_data_trace(p::BosipProblem)
    X = p.problem.data.X
    @assert size(X, 1) == 2
    n = size(X, 2)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="X₁", ylabel="X₂")

    # Create a color gradient from blue to red
    cs = cgrad(:viridis)[range(0, 1; length=n)]
    lines!(ax, X[1,:], X[2,:]; color=cs, label="Data Trace")
    # Scatter the first 3 points with X markers
    scatter!(ax, X[1, 1:3], X[2, 1:3]; marker=:x, color=:black, markersize=12, label="First 3 Points")
    axislegend(ax)
    return fig
end

function plot_acquisition(problem::AbstractProblem, p_::BosipProblem; opt=false)
    p = deepcopy(p_)
    markersize = 10

    acquisition = IMMD(;
        y_samples = 20, # TODO 2 * 10*y_dim(problem)
        x_samples = 200, # TODO 2 * 10^x_dim(problem),
        x_proposal = x_prior(problem),
        y_kernel = BOSS.GaussianKernel(),
        p_kernel = BOSS.GaussianKernel(),
    )
    # acquisition = EIV(
    #     y_samples = 20, # TODO 2 * 10*y_dim(problem)
    #     x_samples = 200, # TODO 2 * 10^x_dim(problem),
    #     x_proposal = x_prior(problem),
    # )
    p.problem.acquisition = BOSIP.AcqWrapper(acquisition, p, BosipOptions())

    if opt
        acq_maximizer = OptimizationAM(;
            algorithm = BOBYQA(),
            multistart = 24,
            parallel = parallel(),
            rhoend = 1e-4,
        )

        t_opt = @elapsed x_, _, safe_acq = maximize_acquisition(acq_maximizer, p.problem, BossOptions())
        @info "Optimized acquisition in $t_opt seconds."
        @info "New point: $x_"
        @assert safe_acq isa BOSS.SafeFunction
        acq = safe_acq.f
    else
        acq = acquisition(p, BosipOptions())
    end

    bounds = p.problem.domain.bounds
    lb, ub = bounds[1], bounds[2]
    @assert length(lb) == 2

    res = 20 # TODO 50
    x = range(lb[1], ub[1], length=res) |> collect
    y = range(lb[2], ub[2], length=res) |> collect
    # Z = [acq([xi, yi]) for xi in x, yi in y]
    Z = zeros(length(x), length(y))
    Threads.@threads for i in eachindex(x)
        for j in eachindex(y)
            Z[i, j] = acq([x[i], y[j]])
        end
    end

    # TODO rem
    @show any(isnan, Z)
    @show any(isinf, Z)
    @show extrema(Z)
    if opt
        min, max = extrema(Z)
        vals = safe_acq.(eachcol(p_.problem.data.X))
        vals_8 = safe_acq.(eachcol(p_.problem.data.X .+ 1e-8))
        vals_4 = safe_acq.(eachcol(p_.problem.data.X .+ 1e-4))
        vals = (vals .- min) ./ (max - min)
        vals_4 = (vals_4 .- min) ./ (max - min)
        vals_8 = (vals_8 .- min) ./ (max - min)
        println("Acquisition values of data points:")
        for (i,x) in enumerate(eachcol(p_.problem.data.X))
            println("$x -> $(vals[i]), $(vals_8[i]), $(vals_4[i])")
        end
    end

    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="$(acquisition |> typeof |> nameof) with $(acquisition.y_samples) speculative y_ samples", xlabelsize=20, ylabelsize=20, titlesize=20, xticklabelsize=16, yticklabelsize=16)
    contourf!(ax, x, y, Z; colormap=:lajolla)

    # data
    X = p.problem.data.X
    scatter!(ax, X[1, :], X[2, :]; color=:olive, marker=:circle, markersize, alpha=0.5, label="Data Points")

    # x samples for the MC integration
    mc_colors = cgrad(:RdPu)[3:7] |> cgrad # 3:7 from 1:9 available    
    ws = acq.ws
    ws_norm = (ws .- minimum(ws)) ./ (maximum(ws) - minimum(ws) + eps()) # avoid div by zero
    cs = mc_colors[ws_norm]
    scatter!(ax, acq.xs[1, :], acq.xs[2, :]; color=cs, marker=:x, markersize=markersize, alpha=1., label="$(acquisition.x_samples) x samples for integration")
    Colorbar(fig[1, 2], colormap=mc_colors, label="MC sample weight", width=20)

    # the next point
    if opt
        @assert length(x_) == 2
        scatter!(ax, [x_[1]], [x_[2]]; color=:red, marker=:star5, markersize, alpha=0.5, label="New Point")
    end

    axislegend(ax, position=:lt)
    display(fig)

    # TODO
    return acq 
end

function plot_vals_between(f, xa, xb)
    n = 100
    ts = range(0, 1, length=n) |> collect
    ys = [f((1 - t) * xa + t * xb) for t in ts]

    fig = Figure()
    name = f |> typeof |> nameof
    ax = Axis(fig[1, 1], xlabel="$xa -> $xb", ylabel="$name")
    lines!(ax, ts, ys)
    return fig
end

function contour_around(f, x, radius; points=nothing)
    n = 50
    xs = range(x[1] - radius, x[1] + radius, length=n) |> collect
    ys = range(x[2] - radius, x[2] + radius, length=n) |> collect
    Z = [f([xi, yi]) for xi in xs, yi in ys]

    # TODO rem
    @show extrema(Z)
    opt_idx = argmax(Z)
    opt_x = [xs[opt_idx[1]], ys[opt_idx[2]]]
    @show opt_x, f(opt_x)

    fig = Figure()
    name = f |> typeof |> nameof
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="$name around $x")
    contourf!(ax, xs, ys, Z)
    
    if !isnothing(points)
        @assert size(points, 1) == 2
        scatter!(ax, points[1, :], points[2, :]; color=:red, marker=:x, markersize=10)
    end

    scatter!(ax, [opt_x[1]], [opt_x[2]]; color=:green, marker=:star5, markersize=14, label="opt_x")

    return fig
end

# TODO remove
function tt(acq)
    x_mmd = [2.565050444417085, 0.354112189423657]
    x_eiv = [2.149682480197693, 0.3486922515959592]

    ws_mmd, vals_mmd = acq(x_mmd)
    ws_eiv, vals_eiv = acq(x_eiv)

    @assert ws_mmd == ws_eiv
    ws = ws_mmd
    n = length(ws)

    # Normalize arrays to [0, 1]
    normalize(v) = (v .- minimum(v)) ./ (maximum(v) - minimum(v))
    ws = normalize(ws)
    ys_mmd = ws .* vals_mmd
    ys_eiv = ws .* vals_eiv
    # ys_mmd = normalize(ys_mmd)
    # ys_eiv = normalize(ys_eiv)

    cumsum_mmd = cumsum(ys_mmd) ./ sum(ys_mmd) .* maximum(ys_mmd)
    cumsum_eiv = cumsum(ys_eiv) ./ sum(ys_eiv) .* maximum(ys_eiv)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Sample Index", ylabel="", title="Acq. computation")
    # scatter!(ax, 1:n, ws; color=:blue, marker=:circle, markersize=8, label="ws")
    scatter!(ax, 1:n, ys_mmd; color=:red, marker=:rect, markersize=8, label="x_mmd values")
    scatter!(ax, 1:n, ys_eiv; color=:green, marker=:utriangle, markersize=8, label="x_eiv values")
    lines!(ax, 1:n, cumsum_mmd; color=:red, linestyle=:dash, label="x_mmd cumsum")
    lines!(ax, 1:n, cumsum_eiv; color=:green, linestyle=:dash, label="x_eiv cumsum")
    axislegend(ax, position=:lt)
    return fig
end

function plot_score(scores::AbstractVector{<:AbstractVector{<:Real}}; xscale=log10)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Iteration", ylabel="Score", title="Score over Iterations", xscale)
    for (i, score) in enumerate(scores)
        lines!(ax, 1:length(score), score, label="Run $i")
    end
    axislegend(ax)
    return fig
end

function plot_lines(vecs...; labels=nothing, xscale=identity, yscale=identity)
    plot_legend = !isnothing(labels)
    isnothing(labels) && (labels = fill("", length(vecs)))
    
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Index", ylabel="Value", title="Multiple Vectors as Lines", xscale=xscale, yscale=yscale)
    
    for (i, vec) in enumerate(vecs)
        lines!(ax, 1:length(vec), vec, label=labels[i])
    end
    
    plot_legend && axislegend(ax; position=:lb)
    return fig
end

function plot_true_posterior(problem::AbstractProblem)
    @assert x_dim(problem) == 2

    ref = reference(problem)
    @assert ref isa Function
    bounds = domain(problem).bounds
    lb, ub = bounds[1], bounds[2]
    res = 100
    x = range(lb[1], ub[1], length=res) |> collect
    y = range(lb[2], ub[2], length=res) |> collect
    Z = [exp(ref([xi, yi])) for xi in x, yi in y]

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="True Posterior")
    contourf!(ax, x, y, Z; colormap=:viridis)
    Colorbar(fig[1, 2], label="Posterior", width=20)
    return fig
end

function plot_model_target(problem::AbstractProblem)
    @assert x_dim(problem) == 2
    @assert y_dim(problem) == 1

    f = true_f(problem)
    bounds = domain(problem).bounds
    lb, ub = bounds[1], bounds[2]
    res = 100
    x = range(lb[1], ub[1], length=res) |> collect
    y = range(lb[2], ub[2], length=res) |> collect
    Z = [f([xi, yi])[1] for xi in x, yi in y]

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="Model Target")
    contourf!(ax, x, y, Z; colormap=:viridis)
    Colorbar(fig[1, 2], label="Target", width=20, limits=extrema(Z))

    return fig
end
