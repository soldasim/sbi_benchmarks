
function pure_sample_posterior(sampler::PureSampler, posterior::Function, domain::Domain, count::Int;
    supersample_ratio = 20,
)
    xs, ws = sample_posterior(sampler, posterior, domain, count)
    @assert allequal(ws)
    return xs
end

function pure_sample_posterior(sampler::WeightedSampler, posterior::Function, domain::Domain, count::Int;
    supersample_ratio = 20,
)
    xs, ws = sample_posterior(sampler, posterior, domain, supersample_ratio * count)
    xs = resample(xs, ws, count)
    return xs
end
