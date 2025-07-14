
"""
    SimpleProblem()

The "simple" problem from Jarvenpaa & Gutmann's "Parallel..." paper.
"""
struct SimpleProblem <: AbstractProblem end


module SimpleProblemModule

import ..SimpleProblem

import ..simulator
import ..domain
import ..y_max
import ..likelihood
import ..prior_mean
import ..x_prior
import ..y_extrema
import ..noise_std_priors
import ..true_f
import ..reference_samples

using BOSS
using BOLFI
using Distributions
using Bijectors

# --- API ---

simulator(::SimpleProblem) = simulation

domain(::SimpleProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::SimpleProblem) = get_likelihood()

prior_mean(::SimpleProblem) = [0.]

x_prior(::SimpleProblem) = get_x_prior()

# TODO
y_extrema(::SimpleProblem) = ([1.], [1000.])

noise_std_priors(::SimpleProblem) = get_noise_std_priors()

true_f(::SimpleProblem) = x -> [f_(x)]


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-16., x_dim()), fill(16., x_dim()))


# - - - OBSERVATION - - - - -

"""observation"""
const z_obs = [0.]
const y_dim = 1

"""simulation noise std"""
const ω = fill(1., y_dim)


# - - - EXPERIMENT - - - - -

const ρ = 0.25
const inv_S = inv([1.; ρ;; ρ; 1.;;])
f_(x) = -(1/2) * x' * inv_S * x

# TODO sqrt
function simulation(x; noise_std=ω)
    y1 = f_(x) + rand(Normal(0., noise_std[1]))
    return [y1]
end
# function simulation(x; noise_std=ω)
#     y1 = f_(x) + rand(Normal(0., noise_std[1]))
#     return [sqrt((-2) * y1)]
# end

# The objective for the GP.
obj(x) = simulation(x)

# TODO sqrt
get_likelihood() = ExpLikelihood()
# get_likelihood() = SqExpLikelihood()


# truncate the prior to the bounds
function get_x_prior()
    return Product(Uniform.(get_bounds()...))
end


# - - - HYPERPARAMETERS - - - - -

# get_acquisition() = PostVarAcq()
get_acquisition() = LogPostVarAcq()
# get_acquisition() = InfoGainInt(;
#         x_samples = 1000,
#         samples = 20,
#         x_proposal = get_x_prior(),
#         y_kernel = BOSS.GaussianKernel(),
#         p_kernel = BOSS.GaussianKernel(),
#     )

get_kernel() = BOSS.ExponentialKernel() # TODO ???

function get_lengthscale_priors()
    ranges = (-1.) * .-(get_bounds()...)

    d = TDist(4)
    d = truncated(d; lower=0.)
    ds = transformed.(Ref(d), Bijectors.Scale.(ranges ./ 2))

    return fill(product_distribution(ds), y_dim)
end

function get_amplitude_priors()
    d = TDist(4)
    d = truncated(d; lower=0.)
    d = transformed(d, Bijectors.Scale(1000.))
    
    return fill(d, y_dim)
end

function get_noise_std_priors()
    d = TDist(4)
    d = truncated(d; lower=0.)
    d = transformed(d, Bijectors.Scale(50.))

    return fill(d, y_dim)
end

get_model() = GaussianProcess(;
    kernel = get_kernel(),
    lengthscale_priors = get_lengthscale_priors(),
    amplitude_priors = get_amplitude_priors(),
    noise_std_priors = get_noise_std_priors(),
)


# - - - INITIAL DATA - - - - -

function get_init_data(count)
    X = rand(get_x_prior(), count)
    Y = reduce(hcat, (obj(x) for x in eachcol(X)))[:,:]
    return BOSS.ExperimentData(X, Y)
end


# - - - INITIALIZATION - - - - -

bolfi_problem(init_data::Int) = bolfi_problem(get_init_data(init_data))

function bolfi_problem(data::ExperimentData)
    return BolfiProblem(data;
        f = obj,
        domain = Domain(; bounds=get_bounds()),
        acquisition = get_acquisition(),
        model = get_model(),
        likelihood = get_likelihood(),
        x_prior = get_x_prior(),
    )
end

function true_post(x)
    ll = true_like(x)
    pθ = pdf(ToyProblem.get_x_prior(), x)
    return pθ * ll
end

function true_like(x)
    y = ToyProblem.simulation(x; noise_std=zeros(ToyProblem.y_dim))
    ll = exp(y[1]) # proportional
    return ll
end

function get_eval_grid()
    iter = Iterators.product(range(-16, 16; length=75), range(-16, 16; length=75))
    xs = mapreduce(x -> [x...], hcat, iter)
    
    lb, ub = get_bounds()
    domain_area = prod(ub .- lb)
    
    # sampling probability of each sample is `1 / domain_area`
    # the weights should be inverse of the sampling probability
    ws = fill(domain_area, size(xs, 2))

    return xs, ws
end

end # module SimpleProblemModule
