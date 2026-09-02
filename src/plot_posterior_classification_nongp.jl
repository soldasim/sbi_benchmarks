"""
Posterior classification scatter plot — NonstationaryGP (nongp) vs Standard comparison,
UNPAIRED median-curve AUC-difference variant.
Plot 5c (extended 2026-08-03 to all 37 problems; previously Group-A-only, n=7).

Corrected 2026-09-01: this docstring (and the caption banner rendered on the
figure itself) previously described a PAIRED metric, but the data source
(src/compute_median_curve_auc_scores.jl, renamed from the misleadingly-named
compute_paired_auc_scores.jl) has been unpaired since 2026-08-05. For the
genuinely paired sibling, see plot_posterior_classification_nongp_paired.jl /
compute_paired_run_auc_scores.jl.

Same layout as plot_posterior_classification_warpedgp.jl, but the colour axis compares
NonstatGP/MaxVar vs Standard(GP)/MaxVar using the UNPAIRED, iteration-normalized,
difference-of-median-curves AUC score from src/compute_median_curve_auc_scores.jl
(2026-08-04) — supersedes the older single-point "final" and "fair" conventions (see
plot_posterior_classification_final.jl's docstring for the full metric definition).

## Why not "final" (see cluster_scripts/notes_paper_plots.md Task 9 for full detail)

nongp (NonstationaryGP) is far more expensive per BO iteration than standard/maxvar, so
it always stops (via partition wall-time) at a much lower iteration count — e.g. Group A:
30-82/101 iters vs standard's clean 101/101; Group B: 39-71/100 vs 101/101; Group C/D:
similarly truncated relative to 200-iter targets. Scoring each run at ITS OWN last
iteration ("final") therefore systematically compares a much-further-converged standard/
maxvar baseline against an early-stopped nongp — exactly the confound documented in
project_bosip_reviews.md (2026-07-15): the original Group-A-only "final" comparison found
Standard winning almost everywhere (including on SIRProblem, the paper's own flagship
sharp-ridge motivating example — directly cutting against the metric's motivating story).
The median-curve AUC metric fixes this by construction (both configs' median curves
integrated over the same shared window T) rather than approximating around it with a
single shared-budget point — it is NOT a wall-clock/compute-cost comparison (nongp costs
much more per iteration than standard); both framings are legitimate for different
questions, but this plot answers the per-iteration modeling-quality question, and says so
on the caption banner.

## DiffusionProblem5D exclusion

6 of DiffusionProblem5D's 20 nongp runs (indices 6,8,10,11,13,16) crashed near-empty
(4-14 iters, PosDefException in ConvergenceCallback — see
project_bosip_benchmarks_nongp_expansion.md) and are EXCLUDED from this problem's
shared-window computation in compute_median_curve_auc_scores.jl (otherwise the shared T would
collapse to ~4 for the whole problem). Flagged again below, not silently dropped.

Data is read dynamically from:
  plots/classify_posteriors.csv  — h_rel, n_modes per problem
  plots/nongp_scores_auc.csv     — nongp_diff, nongp_T per problem (37 rows; run
                                    `include("src/compute_acq_scores_final.jl")` then
                                    `include("src/compute_median_curve_auc_scores.jl")` first if stale)

Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.
Results saved to plots/posterior_classification_nongp.{png,pdf}.
"""

using CairoMakie

include("data_quality.jl")
include("label_placement.jl")

## ── Color scale ──────────────────────────────────────────────────────────────
## Colorbar axis spans the full theoretical range [-1, 1]. Colors sampled from
## :coolwarm (extreme blue/red + midpoint grey + quarter-points for "light"
## shades), each through the same 0.8 darkening factor used everywhere here.
##
## CURRENT (2026-08-05): DISCRETE 5-color step scheme, no smooth blending.
## Bin edges at |diff| = 0.02 and 0.2 — see
## plot_posterior_classification_final.jl's docstring for the full rationale.
##
## PREVIOUS SCHEMES (kept commented for reference/revert, do not delete):
## — 2026-08-05, plain bi-linear, saturating at 0.2:
#   const D_SAT_NG = 0.2
#   function margin_color_ng(diff::Real)
#       d = clamp(Float64(diff), -D_SAT_NG, D_SAT_NG)
#       t = (d + D_SAT_NG) / (2 * D_SAT_NG)
#       return _CMAP_NG[t]
#   end
## — 2026-08-04, warped/saturated at 0.5:
#   const D_SAT_NG        = 0.5
#   const _GREY_SHRINK_NG = 0.25
#   function margin_color_ng(diff::Real)
#       d  = clamp(Float64(diff), -D_SAT_NG, D_SAT_NG)
#       x  = d / D_SAT_NG
#       xw = sign(x) * abs(x)^_GREY_SHRINK_NG
#       t  = (xw + 1) / 2
#       return _CMAP_NG[t]
#   end

const D_MAX_NG   = 1.0
const D_SMALL_NG = 0.01   # below this: draw (grey)
const D_SAT_NG   = 0.1    # at/above this: "fully won" (full red/blue)

_coolwarm_dark_ng(t) = (c = cgrad(:coolwarm)[t]; f = 0.80f0; RGBf(c.r*f, c.g*f, c.b*f))
const _BLUE_NG  = _coolwarm_dark_ng(0.0)
const _LBLUE_NG = _coolwarm_dark_ng(0.25)
const _GREY_NG  = _coolwarm_dark_ng(0.5)
const _LRED_NG  = _coolwarm_dark_ng(0.75)
const _RED_NG   = _coolwarm_dark_ng(1.0)
const _CMAP_NG  = cgrad([_BLUE_NG, _GREY_NG, _RED_NG])   # kept for reference; not used by the discrete scheme

function margin_color_ng(diff::Real)
    d  = Float64(diff)
    ad = abs(d)
    ad < D_SMALL_NG && return _GREY_NG
    ad < D_SAT_NG   && return d > 0 ? _LRED_NG : _LBLUE_NG
    return d > 0 ? _RED_NG : _BLUE_NG
end

## ── Load data ────────────────────────────────────────────────────────────────

function _read_csv_cols_ng(path, cols)
    lines = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

# scores: problem → paired AUC diff (NonstatGP candidate − Standard baseline)
_scr_rows_ng = _read_csv_cols_ng("plots/nongp_scores_auc.csv", ["problem", "nongp_diff"])
_scr_dict_ng = Dict(r[1] => parse(Float64, r[2]) for r in _scr_rows_ng)

# shared window T per problem, for reporting coverage in the caption
_T_rows_ng = _read_csv_cols_ng("plots/nongp_scores_auc.csv", ["problem", "nongp_T"])
_T_dict_ng = Dict(r[1] => parse(Int, r[2]) for r in _T_rows_ng)

# classification: problem → (H_rel, n_modes)
_cls_rows_ng = _read_csv_cols_ng("plots/classify_posteriors.csv",
                                  ["problem", "H_rel", "n_modes"])

## :bip = the 9 "BIP-family" problems (7 orig + 2 HD BIP) — drives marker
## bold/size only. :opt = the 28 optimization-function problems (appendix).
const _BIP_NAMES_NG = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                            "SIRProblem", "DuffingProblem", "DiffusionProblem10",
                            "DuffingProblem5", "DiffusionProblem5D"])

## ONLY the original 7 (main-paper) problems get ALL-CAPS label text — their HD
## variants stay normal-case despite being in _BIP_NAMES_NG above.
const _ORIG7_NAMES_NG = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                              "SIRProblem", "DuffingProblem", "DiffusionProblem10"])

## HD (5D) problems get a FILLED scatter marker; 2D problems get a HOLLOW marker
## — independent of the :bip/:opt split above.
const _HD_PROBLEMS_NG = Set(["DuffingProblem5", "DiffusionProblem5D", "RosenbrockProblem5",
                              "StyblinskiTangProblem5", "MichalewiczProblem5", "SphereProblem5"])

function _display_label_ng(prob)
    prob == "DuffingProblem5"        && return "Duffing5D"
    prob == "DiffusionProblem5D"     && return "Diffusion5D"
    prob == "RosenbrockProblem5"     && return "Rosenbrock5D"
    prob == "StyblinskiTangProblem5" && return "StybTang5D"
    prob == "MichalewiczProblem5"    && return "Michalewicz5D"
    prob == "SphereProblem5"         && return "Sphere5D"
    lbl = replace(prob, r"Problem\d*$" => "")
    lbl = replace(lbl, "ExpandedSchafferF6" => "SchafferF6")
    lbl = replace(lbl, "ExpandedZakharov"   => "Zakharov")
    lbl = replace(lbl, "StyblinskiTang"     => "StybTang")
    lbl = replace(lbl, "GoldsteinPrice"     => "Goldstein")
    lbl = replace(lbl, "ThreeHumpCamel"     => "3HumpCamel")
    return lbl
end

## Tuple: (label_text, H_rel, n_modes, score_diff, grp, is_hd) — label_text is
## already case-transformed (ALL-CAPS for the original 7 only).
const PCLASS_DATA_NG = [
    let grp = r[1] in _BIP_NAMES_NG ? :bip : :opt,
        raw_lbl = _display_label_ng(r[1])
        (r[1] in _ORIG7_NAMES_NG ? uppercase(raw_lbl) : raw_lbl,
         parse(Float64, r[2]),
         parse(Int,     r[3]),
         _scr_dict_ng[r[1]],
         grp,
         r[1] in _HD_PROBLEMS_NG)
    end
    for r in _cls_rows_ng if haskey(_scr_dict_ng, r[1])
]

@info "Plot 5c (nongp, unpaired median-curve AUC): $(length(PCLASS_DATA_NG))/$(length(_cls_rows_ng)) problems have matching nongp_scores_auc.csv rows."

## ── Data-quality auto-check (NaN contamination + run coverage) ────────────────
## Iteration-count mismatch is deliberately NOT checked here — that's exactly what
## the shared-window AUC scoring already corrects for by construction. `nongp_npairs`
## counts run INDICES present in both configs — a legacy of the paired version this
## metric superseded; still a reasonable coverage proxy, just not literally what the
## unpaired score uses.
_qual_rows_ng = _read_csv_cols_ng("plots/nongp_scores_auc.csv",
    ["problem", "nongp_npairs", "nongp_nan", "standard_nan"])

DQ_NOTES_NG = String[]
for r in _qual_rows_ng
    local prob = r[1]
    np, n_nan = parse(Int,r[2]), parse(Int,r[3])
    s_nan = parse(Int, r[4])
    parts = String[]
    (np < 20)  && push!(parts, "$np/20 matched-index runs")
    (n_nan > 0) && push!(parts, "$n_nan NaN in shared window")
    isempty(parts) || push!(DQ_NOTES_NG, "$prob/NonstatGP: " * join(parts, ", "))
    (s_nan > 0) && push!(DQ_NOTES_NG, "$prob/Standard: $s_nan NaN in shared window")
end
sort!(DQ_NOTES_NG)

## ── Automatic label placement ─────────────────────────────────────────────────
## CURRENT (2026-08-06): shared greedy candidate-position placer from
## label_placement.jl — see plot_posterior_classification_final.jl's comment
## for the full rationale and the old force-directed version kept commented there.

## Screen-space positions of the quadrant corner annotations, so labels avoid them too.
_QUAD_ANCHORS_NG = [
    (0.015, 0.63,  :left,  :bottom),
    (0.985, 0.63,  :right, :bottom),
    (0.015, 44.0,  :left,  :top),
    (0.985, 44.0,  :right, :top),
]
_quad_d2s_ng(h, m; yrange = (log10(0.60), log10(50.0)), fig_w = 620.0, fig_h = 380.0) =
    ((h - 0.0) / 1.0 * fig_w, (log10(m) - yrange[1]) / (yrange[2] - yrange[1]) * fig_h)
_QUAD_OBSTACLES_NG = [
    (px, py, 42.0)
    for (hc, mc, ha, va) in _QUAD_ANCHORS_NG
    for (px, py) in (_quad_d2s_ng(hc, mc),)
]

_lox_ng, _loy_ng = auto_label_offsets(PCLASS_DATA_NG;
    xrange = (0.0, 1.0), yrange = (log10(0.60), log10(50.0)), fig_w = 620.0, fig_h = 380.0,
    fontsize_of = it -> it[5] == :bip ? 9.0 : 8.0,
    bold_of     = it -> it[5] == :bip,
    obstacles   = _QUAD_OBSTACLES_NG,
)

## ── Figure ───────────────────────────────────────────────────────────────────

mkpath("plots")

@assert all(m <= 50 for (_, _, m, _, _, _) in PCLASS_DATA_NG) "A problem exceeds the y-axis limit of 50 modes"

fig_ng = Figure(; size = (800, 500))

ax_ng = Axis(fig_ng[2, 1];
    xlabel      = "Normalised posterior entropy",
    ylabel      = "Mode count",
    yscale      = log10,
    xticks      = 0.0:0.1:1.0,
    yticks      = ([1, 2, 5, 10, 20, 50], ["1","2","5","10","20","50"]),
    xgridcolor       = (:black, 0.08),
    ygridcolor       = (:black, 0.08),
    backgroundcolor  = :white,
)
xlims!(ax_ng, 0.0, 1.0)
ylims!(ax_ng, 0.60, 50)

## Threshold lines
vlines!(ax_ng, [0.50]; color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)
hlines!(ax_ng, [1.5];  color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)

## Quadrant annotations — corners
for (hc, ha, mc, va, lbl) in [
        (0.015, :left,  0.63, :bottom, "compact\nunimodal"),
        (0.985, :right, 0.63, :bottom, "diffuse\nunimodal"),
        (0.015, :left,  44.0, :top,    "compact\nmultimodal"),
        (0.985, :right, 44.0, :top,    "diffuse\nmultimodal"),
    ]
    text!(ax_ng, hc, mc; text=lbl, color=(RGBf(0.05, 0.20, 0.05), 0.5), fontsize=10,
          align=(ha, va), font=:italic)
end

## Scatter + labels — marker fill: HD (5D) → filled; 2D → hollow.
for (i, (lbl, h, m, d, grp, is_hd)) in enumerate(PCLASS_DATA_NG)
    col = margin_color_ng(d)
    mf  = Float64(m)

    if is_hd
        scatter!(ax_ng, [h], [mf];
            color       = col,
            markersize  = 11,
            strokewidth = 0.8,
            strokecolor = (:black, 0.35),
        )
    else
        scatter!(ax_ng, [h], [mf];
            color       = (:white, 1.0),
            markersize  = 7,
            strokewidth = 1.5,
            strokecolor = col,
        )
    end

    text!(ax_ng, h, mf;
        text     = lbl,
        color    = col,
        fontsize = grp == :bip ? 9.0 : 8.0,
        font     = grp == :bip ? :bold : :regular,
        align    = (:center, :center),
        offset   = Point2f(_lox_ng[i], _loy_ng[i]),
    )
end

## Colorbar — resample margin_color_ng itself (not the raw 3-color _CMAP_NG) so
## the displayed bands show the discrete step scheme.
_cmap_display_ng = let n = 256
    cgrad([margin_color_ng(-D_MAX_NG + 2*D_MAX_NG*(i-1)/(n-1)) for i in 1:n])
end
Colorbar(fig_ng[2, 2];
    colormap   = _cmap_display_ng,
    limits     = (-D_MAX_NG, D_MAX_NG),
    ticks      = ([-D_MAX_NG, -D_SAT_NG, -D_SMALL_NG, 0, D_SMALL_NG, D_SAT_NG, D_MAX_NG],
                  ["-1\n(NonstatGP wins)", "-$(D_SAT_NG)", "-$(D_SMALL_NG)", "0", "$(D_SMALL_NG)", "$(D_SAT_NG)", "+1\n(Standard wins)"]),
    label      = "Score margin: NonstatGP − Standard  (unpaired median-curve AUC diff, shared window, range [-1,1])",
    ticklabelsize = 11,
    labelsize  = 12,
    labelpadding = -50,
    width      = 14,
    tellheight = false,
)

colsize!(fig_ng.layout, 1, Relative(0.85))

## Caption banner — this is an UNPAIRED, shared-window, difference-of-median-curves
## AUC comparison (each config's runs pooled into one median curve, NOT matched by
## run index; difference of the two curves' mean-TV-in-window, bounded in [-1,1]),
## and the shared windows are far short of the paper's iteration targets for every
## problem (nongp is much more expensive per iteration).
_T_min_ng, _T_max_ng = extrema(values(_T_dict_ng))
_banner_txt_ng = "READ BEFORE CITING: unpaired median-curve AUC-diff over the shared iteration window (not \"final\"). " *
    "Shared windows range $(_T_min_ng)-$(_T_max_ng) iters, all far short of the 100/200-iter " *
    "paper targets (nongp is far more expensive per iteration). DiffusionProblem5D excludes " *
    "6/20 crashed nongp runs (idx 6,8,10,11,13,16) from its shared window. Does NOT represent " *
    "a wall-clock/compute-cost comparison."
if !isempty(DQ_NOTES_NG)
    shown = DQ_NOTES_NG[1:min(length(DQ_NOTES_NG), 4)]
    _banner_txt_ng *= "  |  DATA QUALITY: " * join(shown, "; ")
    (length(DQ_NOTES_NG) > 4) && (_banner_txt_ng *= "  (+$(length(DQ_NOTES_NG)-4) more)")
end
Label(fig_ng[1, :], _banner_txt_ng;
    fontsize = 12, color = :firebrick, tellwidth = false)

rowsize!(fig_ng.layout, 1, Auto(0.12))

## Save
save("plots/posterior_classification_nongp.png", fig_ng; px_per_unit = 3)
save("plots/posterior_classification_nongp.pdf", fig_ng)
@info "Saved → plots/posterior_classification_nongp.{png,pdf}"
