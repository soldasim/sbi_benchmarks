"""
Posterior classification scatter plot — WarpedGP vs Standard comparison,
UNPAIRED median-curve AUC-difference variant.

Corrected 2026-09-01: this docstring previously described a PAIRED metric, but
the data source (src/compute_median_curve_auc_scores.jl, renamed from the
misleadingly-named compute_paired_auc_scores.jl) has been unpaired since
2026-08-05. For the genuinely paired sibling, see
plot_posterior_classification_warpedgp_paired.jl / compute_paired_run_auc_scores.jl.

Same layout as plot_posterior_classification_final.jl, but the colour axis
compares WarpedGP YJA/MaxVar vs Standard GP/MaxVar using the UNPAIRED,
iteration-normalized, difference-of-median-curves AUC score from
src/compute_median_curve_auc_scores.jl (see that file's docstring, or
plot_posterior_classification_final.jl's, for the full definition — each
config's runs pooled into one median TV-curve, NOT matched by run index,
difference of the two curves' shared-window AUC, bounded in [-1,1]).
Color encodes: diff = warpedgp_diff (candidate=WarpedGP, baseline=Standard)
  < 0 → WarpedGP wins (cool/blue), > 0 → Standard wins (warm/red)

Data is read dynamically from:
  plots/classify_posteriors.csv   — h_rel, n_modes per problem
  plots/warpedgp_scores_auc.csv   — warpedgp_diff per problem

Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.
Results saved to plots/posterior_classification_warpedgp.{png,pdf}.
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
#   const D_SAT_WG = 0.2
#   function margin_color_wg(diff::Real)
#       d = clamp(Float64(diff), -D_SAT_WG, D_SAT_WG)
#       t = (d + D_SAT_WG) / (2 * D_SAT_WG)
#       return _CMAP_WG[t]
#   end
## — 2026-08-04, warped/saturated at 0.5:
#   const D_SAT_WG        = 0.5
#   const _GREY_SHRINK_WG = 0.25
#   function margin_color_wg(diff::Real)
#       d  = clamp(Float64(diff), -D_SAT_WG, D_SAT_WG)
#       x  = d / D_SAT_WG
#       xw = sign(x) * abs(x)^_GREY_SHRINK_WG
#       t  = (xw + 1) / 2
#       return _CMAP_WG[t]
#   end

const D_MAX_WG   = 1.0
const D_SMALL_WG = 0.01   # below this: draw (grey)
const D_SAT_WG   = 0.1    # at/above this: "fully won" (full red/blue)

_coolwarm_dark_wg(t) = (c = cgrad(:coolwarm)[t]; f = 0.80f0; RGBf(c.r*f, c.g*f, c.b*f))
const _BLUE_WG  = _coolwarm_dark_wg(0.0)
const _LBLUE_WG = _coolwarm_dark_wg(0.25)
const _GREY_WG  = _coolwarm_dark_wg(0.5)
const _LRED_WG  = _coolwarm_dark_wg(0.75)
const _RED_WG   = _coolwarm_dark_wg(1.0)
const _CMAP_WG  = cgrad([_BLUE_WG, _GREY_WG, _RED_WG])   # kept for reference; not used by the discrete scheme

function margin_color_wg(diff::Real)
    d  = Float64(diff)
    ad = abs(d)
    ad < D_SMALL_WG && return _GREY_WG
    ad < D_SAT_WG   && return d > 0 ? _LRED_WG : _LBLUE_WG
    return d > 0 ? _RED_WG : _BLUE_WG
end

## ── Load data ────────────────────────────────────────────────────────────────

function _read_csv_cols_wg(path, cols)
    lines = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

# scores: problem → paired AUC diff (WarpedGP candidate − Standard baseline)
_scr_rows_wg = _read_csv_cols_wg("plots/warpedgp_scores_auc.csv", ["problem", "warpedgp_diff"])
_scr_dict_wg = Dict(r[1] => parse(Float64, r[2]) for r in _scr_rows_wg)

# classification: problem → (H_rel, n_modes)
_cls_rows_wg = _read_csv_cols_wg("plots/classify_posteriors.csv",
                                  ["problem", "H_rel", "n_modes"])

## :bip = the 9 "BIP-family" problems (7 orig + 2 HD BIP) — drives marker
## bold/size only. :opt = the 28 optimization-function problems (appendix).
const _BIP_NAMES_WG = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                            "SIRProblem", "DuffingProblem", "DiffusionProblem10",
                            "DuffingProblem5", "DiffusionProblem5D"])

## ONLY the original 7 (main-paper) problems get ALL-CAPS label text — their HD
## variants stay normal-case despite being in _BIP_NAMES_WG above.
const _ORIG7_NAMES_WG = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                              "SIRProblem", "DuffingProblem", "DiffusionProblem10"])

## HD (5D) problems get a FILLED scatter marker; 2D problems get a HOLLOW marker
## — independent of the :bip/:opt split above.
const _HD_PROBLEMS_WG = Set(["DuffingProblem5", "DiffusionProblem5D", "RosenbrockProblem5",
                              "StyblinskiTangProblem5", "MichalewiczProblem5", "SphereProblem5"])

function _display_label_wg(prob)
    # 5D problems get a "5D" suffix to distinguish from their 2D counterparts
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
const PCLASS_DATA_WG = [
    let grp = r[1] in _BIP_NAMES_WG ? :bip : :opt,
        raw_lbl = _display_label_wg(r[1])
        (r[1] in _ORIG7_NAMES_WG ? uppercase(raw_lbl) : raw_lbl,
         parse(Float64, r[2]),
         parse(Int,     r[3]),
         _scr_dict_wg[r[1]],
         grp,
         r[1] in _HD_PROBLEMS_WG)
    end
    for r in _cls_rows_wg if haskey(_scr_dict_wg, r[1])
]

## ── Data-quality auto-check (NaN contamination / run coverage) ────────────────
## The median-curve AUC metric already neutralizes unequal iteration budgets by
## construction — only missing-run coverage and in-window NaN remain worth
## flagging. `warpedgp_npairs` counts run INDICES present in both configs — a
## legacy of the paired version this metric superseded; still a reasonable
## coverage proxy, just not literally what the unpaired score uses.
_qual_rows_wg = _read_csv_cols_wg("plots/warpedgp_scores_auc.csv", ["problem", "warpedgp_npairs", "warpedgp_nan", "standard_nan"])

DQ_NOTES_WG = String[]
for r in _qual_rows_wg
    local prob = r[1]
    np, nn = parse(Int,r[2]), parse(Int,r[3])
    s_nan = parse(Int, r[4])
    parts = String[]
    (np < 20) && push!(parts, "$np/20 matched-index runs")
    (nn > 0)  && push!(parts, "$nn NaN in shared window")
    isempty(parts) || push!(DQ_NOTES_WG, "$prob/WarpedGP: " * join(parts, ", "))
    (s_nan > 0) && push!(DQ_NOTES_WG, "$prob/Standard: $s_nan NaN in shared window")
end
sort!(DQ_NOTES_WG)

## ── Automatic label placement ─────────────────────────────────────────────────
## CURRENT (2026-08-06): shared greedy candidate-position placer from
## label_placement.jl — see plot_posterior_classification_final.jl's comment
## for the full rationale and the old force-directed version kept commented there.

## Screen-space positions of the quadrant corner annotations, so labels avoid them too.
_QUAD_ANCHORS_WG = [
    (0.015, 0.63,  :left,  :bottom),
    (0.985, 0.63,  :right, :bottom),
    (0.015, 44.0,  :left,  :top),
    (0.985, 44.0,  :right, :top),
]
_quad_d2s_wg(h, m; yrange = (log10(0.60), log10(50.0)), fig_w = 620.0, fig_h = 380.0) =
    ((h - 0.0) / 1.0 * fig_w, (log10(m) - yrange[1]) / (yrange[2] - yrange[1]) * fig_h)
_QUAD_OBSTACLES_WG = [
    (px, py, 42.0)
    for (hc, mc, ha, va) in _QUAD_ANCHORS_WG
    for (px, py) in (_quad_d2s_wg(hc, mc),)
]

_lox_wg, _loy_wg = auto_label_offsets(PCLASS_DATA_WG;
    xrange = (0.0, 1.0), yrange = (log10(0.60), log10(50.0)), fig_w = 620.0, fig_h = 380.0,
    fontsize_of = it -> it[5] == :bip ? 9.0 : 8.0,
    bold_of     = it -> it[5] == :bip,
    obstacles   = _QUAD_OBSTACLES_WG,
)

## ── Figure ───────────────────────────────────────────────────────────────────

mkpath("plots")

@assert all(m <= 50 for (_, _, m, _, _, _) in PCLASS_DATA_WG) "A problem exceeds the y-axis limit of 50 modes"

fig_wg = Figure(; size = (800, 460))

ax_wg = Axis(fig_wg[1, 1];
    xlabel      = "Normalised posterior entropy",
    ylabel      = "Mode count",
    yscale      = log10,
    xticks      = 0.0:0.1:1.0,
    yticks      = ([1, 2, 5, 10, 20, 50], ["1","2","5","10","20","50"]),
    xgridcolor       = (:black, 0.08),
    ygridcolor       = (:black, 0.08),
    backgroundcolor  = :white,
)
xlims!(ax_wg, 0.0, 1.0)
ylims!(ax_wg, 0.60, 50)

## Threshold lines
vlines!(ax_wg, [0.50]; color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)
hlines!(ax_wg, [1.5];  color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)

## Quadrant annotations — corners
for (hc, ha, mc, va, lbl) in [
        (0.015, :left,  0.63, :bottom, "compact\nunimodal"),
        (0.985, :right, 0.63, :bottom, "diffuse\nunimodal"),
        (0.015, :left,  44.0, :top,    "compact\nmultimodal"),
        (0.985, :right, 44.0, :top,    "diffuse\nmultimodal"),
    ]
    text!(ax_wg, hc, mc; text=lbl, color=(RGBf(0.05, 0.20, 0.05), 0.5), fontsize=10,
          align=(ha, va), font=:italic)
end

## Scatter + labels — marker fill: HD (5D) → filled; 2D → hollow.
for (i, (lbl, h, m, d, grp, is_hd)) in enumerate(PCLASS_DATA_WG)
    col = margin_color_wg(d)
    mf  = Float64(m)

    if is_hd
        scatter!(ax_wg, [h], [mf];
            color       = col,
            markersize  = 11,
            strokewidth = 0.8,
            strokecolor = (:black, 0.35),
        )
    else
        scatter!(ax_wg, [h], [mf];
            color       = (:white, 1.0),
            markersize  = 7,
            strokewidth = 1.5,
            strokecolor = col,
        )
    end

    text!(ax_wg, h, mf;
        text     = lbl,
        color    = col,
        fontsize = grp == :bip ? 9.0 : 8.0,
        font     = grp == :bip ? :bold : :regular,
        align    = (:center, :center),
        offset   = Point2f(_lox_wg[i], _loy_wg[i]),
    )
end

## Colorbar — resample margin_color_wg itself (not the raw 3-color _CMAP_WG) so
## the displayed bands show the discrete step scheme.
_cmap_display_wg = let n = 256
    cgrad([margin_color_wg(-D_MAX_WG + 2*D_MAX_WG*(i-1)/(n-1)) for i in 1:n])
end
Colorbar(fig_wg[1, 2];
    colormap   = _cmap_display_wg,
    limits     = (-D_MAX_WG, D_MAX_WG),
    ticks      = ([-D_MAX_WG, -D_SAT_WG, -D_SMALL_WG, 0, D_SMALL_WG, D_SAT_WG, D_MAX_WG],
                  ["-1\n(WarpedGP wins)", "-$(D_SAT_WG)", "-$(D_SMALL_WG)", "0", "$(D_SMALL_WG)", "$(D_SAT_WG)", "+1\n(Standard wins)"]),
    label      = "Score margin: WarpedGP − Standard  (unpaired median-curve AUC diff, shared window, range [-1,1])",
    ticklabelsize = 11,
    labelsize  = 12,
    labelpadding = -50,
    width      = 14,
    tellheight = false,
)

colsize!(fig_wg.layout, 1, Relative(0.85))

dq_add_banner!(fig_wg, DQ_NOTES_WG; max_shown=6, fontsize=10)

## Save
save("plots/posterior_classification_warpedgp.png", fig_wg; px_per_unit = 3)
save("plots/posterior_classification_warpedgp.pdf", fig_wg)
@info "Saved → plots/posterior_classification_warpedgp.{png,pdf}"
