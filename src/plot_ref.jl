include("main.jl")

function compute_all()
    for problem in [
        ABProblem(),
        SimpleProblem(),
        BananaProblem(),
        BimodalProblem(),
        SIRProblem(),
        DuffingProblem(),
        DiffusionProblem(),
    ]
        @info "Computing marginals for $(typeof(problem))"
        @time compute_marginals(problem; save=true)
    end
end

function plot_all()
    problems = [ABProblem(), SimpleProblem(), BananaProblem(), BimodalProblem(), SIRProblem(), DuffingProblem(), DiffusionProblem()]
    fontsizes = [20, 20, 20, 20, 15, 15, 15]

    for (prob, font) in zip(problems, fontsizes) 
        plot_marginals(prob; save=true, display=true, base_fontsize=font)
    end
end

param_labels(p::AbstractProblem) = ["x$i" for i in 1:x_dim(p)]
param_labels(::ABProblem) = ["a", "b"]
param_labels(::SIRProblem) = ["β", "γ"]
param_labels(::DuffingProblem) = ["δ", "α", "β"]
param_labels(::DiffusionProblem) = ["xₛ", "yₛ", "tₛ"]

function save_data(problem::AbstractProblem, data::PlotData)
    @save "data/marginals/" * string(typeof(problem))[1:end-7] * "_marginals.jld2" data
end
function load_data(problem::AbstractProblem)
    data = nothing
    @load "data/marginals/" * string(typeof(problem))[1:end-7] * "_marginals.jld2" data
    return data
end

_plot_settings(problem::AbstractProblem) = PlotSettings(;
        grid_size = 50, # TODO 200
        resolution = 400,
        param_labels = param_labels(problem),
        plot_data = false,
        full_matrix = false,
        plot_diagonal = false,
        # x_true = get_x_ref(problem),
        title = string(typeof(problem))[1:end-7],
)

function compute_marginals(problem::AbstractProblem; save=true)
    ref = reference(problem)
    @assert ref isa Function
    ref_post = x -> exp.(ref(x))
    # ref_post = ref

    # just a dummy to call plot_marginals_int
    p = construct_bosip_problem(;
        problem,
        data = get_init_data(problem, 3),
        acquisition = LogMaxVar(),
        model = GaussianProcess(;
            mean = prior_mean(problem),
            kernel = BOSS.Matern52Kernel(),
            lengthscale_priors = get_lengthscale_priors(problem),
            amplitude_priors = get_amplitude_priors(problem),
            noise_std_priors = get_noise_std_priors(problem),
        ),
    )

    plot_settings = _plot_settings(problem)

    data = compute_marginals_int(p;
        func = (p) -> ref_post,
        lhc_grid_size = 50 * 10^(x_dim(problem) - 2), # TODO 50 * 10^(x_dim(problem) - 2)
        plot_settings,
    )

    save && save_data(problem, data)
    return data
end

function plot_marginals(problem::AbstractProblem, data = nothing; save=false, display=false, base_fontsize=20)
    set_theme_fonts!(; base_fontsize)

    isnothing(data) && (data = load_data(problem))
    
    plot_settings = _plot_settings(problem)
    fig = BOSIP.plot_marginals(data; plot_settings)

    save && CairoMakie.save("plots/marginals/post_" * string(typeof(problem))[1:end-7] * ".pdf", fig)
    display && CairoMakie.display(fig)
    return fig
end

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
