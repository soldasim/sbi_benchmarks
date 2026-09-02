"""
Metadata-based candidate metrics (2026-08-04): cheap, no simulation needed.

  - `snr` — est_amplitude / est_noise_std (elementwise ratio, max over output
    dims). Noiseless problems (est_noise_std === nothing) get a large sentinel
    SNR (1e4) representing "no observation noise".
  - `is_bip` — 1.0 for the 7 original BIP problems + 2 HD BIP, 0.0 for the
    opt-function-derived (cross-polytope) problems. Crude categorical check:
    is one problem family just systematically different regardless of any
    shape metric?

Must be run after include("src/main.jl"); include("src/classify_smoothness.jl")
(for the problem list / _sm_display_name).

Usage:
    include("src/classify_metadata.jl")
    run_classify_metadata()
"""

const _SNR_NOISELESS_SENTINEL = 1.0e4

const _META_BIP_NAMES = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                              "SIRProblem", "DuffingProblem", "DiffusionProblem10",
                              "DuffingProblem5", "DiffusionProblem5D"])

function _snr(problem)
    amp   = est_amplitude(problem)
    noise = est_noise_std(problem)
    noise === nothing && return _SNR_NOISELESS_SENTINEL
    return maximum(a / max(n, 1e-12) for (a, n) in zip(amp, noise))
end

_is_bip(name) = name in _META_BIP_NAMES ? 1.0 : 0.0

function run_classify_metadata()
    results = NamedTuple[]
    for p in _SM_ALL_PROBLEMS
        name = _sm_display_name(p)
        snr  = _snr(p)
        bip  = _is_bip(name)
        push!(results, (name=name, snr=snr, is_bip=bip))
        println("$name: snr=$(round(snr,digits=3)) is_bip=$bip")
    end
    mkpath("plots")
    open("plots/classify_metadata.csv", "w") do io
        println(io, "problem,snr,is_bip")
        for r in results
            println(io, "$(r.name),$(r.snr),$(r.is_bip)")
        end
    end
    println("\nSaved → plots/classify_metadata.csv")
    return results
end
