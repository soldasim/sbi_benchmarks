"""
Posterior classification scatter plot — UNPAIRED, median-curve AUC-difference
variant.

Corrected 2026-09-01: this docstring previously described a PAIRED metric
(median of per-run differences), but the underlying data source has actually
been unpaired since 2026-08-05 — see src/compute_median_curve_auc_scores.jl
(renamed from the misleadingly-named compute_paired_auc_scores.jl). This file
now describes what it actually plots. For the genuinely paired sibling, see
plot_posterior_classification_paired.jl / compute_paired_run_auc_scores.jl.

Same layout as plot_posterior_classification_loglog.jl, but the colour axis uses
the UNPAIRED, iteration-normalized, difference-of-median-curves AUC score
computed by src/compute_median_curve_auc_scores.jl:

  For each problem: build one median TV-curve per config (median, at each
  iteration t, across ALL of that config's runs — run indices are NOT matched
  between configs), integrate each curve over log-iteration up to the shared
  window T (T = shortest raw score-array length over all loaded runs of BOTH
  configs) to get a mean-TV-in-window value bounded in [0,1] per config, and
  take candidate − baseline. Bounded in [-1,1] by construction.

This supersedes the old "final" (each run's own last iteration) convention,
which could badly mislabel a problem whenever the two configs ran to very
different iteration counts — see the 2026-08-04 ExpandedZakharovProblem2
incident (EIV looked like a decisive MaxVar loss under "final" scoring — diff
0.84 in log-TV terms — purely because most EIV runs stopped around iteration
110/200 while MaxVar ran the full 200; under this shared-window AUC metric the
real difference is 0.0094, a clear draw).

Data is read dynamically from:
  plots/classify_posteriors.csv   — h_rel, n_modes per problem
  plots/acq_scores_auc.csv        — eiv_diff (candidate=EIV, baseline=MaxVar) per problem

Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.
Results saved to plots/posterior_classification_final.{png,pdf}.
"""

using CairoMakie

include("data_quality.jl")
include("label_placement.jl")

## ── Color scale ──────────────────────────────────────────────────────────────
## Colorbar axis spans the full theoretical range [-1, 1]. Colors sampled from
## :coolwarm (extreme blue/red + midpoint grey + the quarter-points for the
## "light" shades), each through the same 0.8 darkening factor previously
## applied everywhere in this plot.
##
## CURRENT (2026-08-05): DISCRETE 5-color step scheme, no smooth blending.
## Bin edges at |diff| = 0.02 and 0.2:
##   |diff| < 0.02        → grey   (draw)
##   0.02 <= |diff| < 0.2 → light red/blue (small edge)
##   |diff| >= 0.2        → full red/blue  (decisive win, "= fully won")
##
## PREVIOUS SCHEMES (kept commented for reference/revert, do not delete):
## — 2026-08-05, plain bi-linear, saturating at 0.2:
#   const D_SAT_F = 0.2
#   function margin_color_f(diff::Real)
#       d = clamp(Float64(diff), -D_SAT_F, D_SAT_F)
#       t = (d + D_SAT_F) / (2 * D_SAT_F)
#       return _CMAP_F[t]
#   end
## — 2026-08-04, warped/saturated at 0.5:
#   const D_SAT_F         = 0.5
#   const _GREY_SHRINK_F  = 0.25
#   function margin_color_f(diff::Real)
#       d  = clamp(Float64(diff), -D_SAT_F, D_SAT_F)
#       x  = d / D_SAT_F
#       xw = sign(x) * abs(x)^_GREY_SHRINK_F
#       t  = (xw + 1) / 2
#       return _CMAP_F[t]
#   end

const D_MAX_F  = 1.0
const D_SMALL_F = 0.01   # below this: draw (grey)
const D_SAT_F   = 0.1    # at/above this: "fully won" (full red/blue)

_coolwarm_dark_f(t) = (c = cgrad(:coolwarm)[t]; f = 0.80f0; RGBf(c.r*f, c.g*f, c.b*f))
const _BLUE_F  = _coolwarm_dark_f(0.0)
const _LBLUE_F = _coolwarm_dark_f(0.25)
const _GREY_F  = _coolwarm_dark_f(0.5)
const _LRED_F  = _coolwarm_dark_f(0.75)
const _RED_F   = _coolwarm_dark_f(1.0)
const _CMAP_F  = cgrad([_BLUE_F, _GREY_F, _RED_F])   # kept for reference; not used by the discrete scheme

function margin_color_f(diff::Real)
    d  = Float64(diff)
    ad = abs(d)
    ad < D_SMALL_F && return _GREY_F
    ad < D_SAT_F   && return d > 0 ? _LRED_F : _LBLUE_F
    return d > 0 ? _RED_F : _BLUE_F
end


## ── Load data ────────────────────────────────────────────────────────────────

function _read_csv_cols_f(path, cols)
    lines = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

# scores: problem → paired AUC diff (EIV candidate − MaxVar baseline)
_scr_rows_f  = _read_csv_cols_f("plots/acq_scores_auc.csv", ["problem", "eiv_diff"])
_scr_dict_f  = Dict(r[1] => parse(Float64, r[2]) for r in _scr_rows_f)

# classification: problem → (H_rel, n_modes)
_cls_rows_f  = _read_csv_cols_f("plots/classify_posteriors.csv",
                                 ["problem", "H_rel", "n_modes"])

## :bip = the 9 "BIP-family" problems (7 orig + 2 HD BIP) — drives marker
## bold/size only. :opt = the 28 optimization-function problems (appendix).
const _BIP_NAMES_F = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                           "SIRProblem", "DuffingProblem", "DiffusionProblem10",
                           "DuffingProblem5", "DiffusionProblem5D"])

## ONLY the original 7 (main-paper) problems get ALL-CAPS label text — their HD
## variants (DuffingProblem5, DiffusionProblem5D) stay normal-case despite being
## in _BIP_NAMES_F above.
const _ORIG7_NAMES_F = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                             "SIRProblem", "DuffingProblem", "DiffusionProblem10"])

## HD (5D) problems get a FILLED scatter marker; 2D problems get a HOLLOW
## (white-fill, colored-outline) marker — independent of the :bip/:opt split
## above (e.g. DuffingProblem5 is :bip AND hd; RosenbrockProblem5 is :opt AND hd).
const _HD_PROBLEMS_F = Set(["DuffingProblem5", "DiffusionProblem5D", "RosenbrockProblem5",
                             "StyblinskiTangProblem5", "MichalewiczProblem5", "SphereProblem5"])

function _display_label_f(prob)
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

## Tuple: (label_text, H_rel, n_modes, score_diff, grp, is_hd)
## label_text is already case-transformed here (ALL-CAPS for the original 7
## only) so both the rendered text!() call and the label-placement box-size
## estimate see the actual displayed string.
const PCLASS_DATA_F = [
    let grp = r[1] in _BIP_NAMES_F ? :bip : :opt,
        raw_lbl = _display_label_f(r[1])
        (r[1] in _ORIG7_NAMES_F ? uppercase(raw_lbl) : raw_lbl,
         parse(Float64, r[2]),
         parse(Int,     r[3]),
         _scr_dict_f[r[1]],
         grp,
         r[1] in _HD_PROBLEMS_F)
    end
    for r in _cls_rows_f if haskey(_scr_dict_f, r[1])
]

## ── Data-quality auto-check (NaN contamination / run coverage) ────────────────
## The median-curve AUC metric already neutralizes unequal iteration budgets by
## construction (both configs' median curves are integrated over the same
## shared window T), so the only remaining things worth flagging are: fewer
## runs available than expected (a config missing files entirely — this makes
## each config's own median curve less representative, even though there's no
## per-run pairing to break), and NaN contamination within the shared window
## (which can bias a config's median curve at those iterations even though T
## itself is fair). `eiv_npairs`/`immd_npairs` below count run INDICES present
## in both configs — a legacy of the paired version this metric superseded;
## still a reasonable coverage proxy, just not literally what the score uses.
_qual_rows_f = _read_csv_cols_f("plots/acq_scores_auc.csv",
    ["problem", "eiv_npairs", "eiv_nan", "immd_npairs", "immd_nan", "maxvar_nan"])

## Group C/D (5D problems) only ever run 5 EIV/IMMD seeds by design (vs 20 for MaxVar
## and for every Group A/B problem) — use the right expected count, not a blanket 20.
## (_HD_PROBLEMS_F itself is defined earlier, alongside _BIP_NAMES_F.)

DQ_NOTES_F = String[]
for r in _qual_rows_f
    local prob = r[1]
    ei_np, ei_nan = parse(Int,r[2]), parse(Int,r[3])
    im_np, im_nan = parse(Int,r[4]), parse(Int,r[5])
    mv_nan = parse(Int, r[6])
    exp_eiv_immd = prob in _HD_PROBLEMS_F ? 5 : 20
    for (lbl, np, nn, exp) in (("EIV",ei_np,ei_nan,exp_eiv_immd), ("IMMD",im_np,im_nan,exp_eiv_immd))
        parts = String[]
        (np < exp) && push!(parts, "$np/$exp matched-index runs")
        (nn > 0)   && push!(parts, "$nn NaN in shared window")
        isempty(parts) || push!(DQ_NOTES_F, "$prob/$lbl: " * join(parts, ", "))
    end
    (mv_nan > 0) && push!(DQ_NOTES_F, "$prob/MaxVar: $mv_nan NaN in shared window")
end
sort!(DQ_NOTES_F)

## ── Automatic label placement ─────────────────────────────────────────────────
## CURRENT (2026-08-06): shared greedy candidate-position placer from
## label_placement.jl — replaces the old force-directed/spring relaxation below
## (kept commented for reference/revert), which let labels drift far from their
## point on a crowded 37-point canvas. The greedy placer always picks the
## nearest non-overlapping candidate position, so labels stay visibly anchored
## to their scatter point.
##
## PREVIOUS (force-directed/spring relaxation) — kept commented for reference/revert:
#   function _auto_label_offsets_f(items;
#       xrange = (0.0, 1.0), yrange = (log10(0.60), log10(50.0)),
#       fig_w = 620.0, fig_h = 380.0, n_iter = 300, lr = 0.25, repel = 50.0,
#       spring_k = 0.012, dead_zone = 10.0, obstacles = Tuple{Float64,Float64}[])
#       n      = length(items)
#       labels = [x[1] for x in items]
#       is_bip = [x[5] == :bip for x in items]
#       fs     = [b ? 9.0 : 8.0 for b in is_bip]
#       hlw    = [length(labels[i]) * fs[i] * 0.55 / 2 for i in 1:n]
#       hlh    = [fs[i] * 0.6 for i in 1:n]
#       function d2s(h, m)
#           x = (h - xrange[1]) / (xrange[2] - xrange[1]) * fig_w
#           y = (log10(Float64(m)) - yrange[1]) / (yrange[2] - yrange[1]) * fig_h
#           (x, y)
#       end
#       pts = [d2s(x[2], x[3]) for x in items]
#       obs_repel = 60.0
#       angles = LinRange(0.0, 2π - 2π/n, n)
#       ox = 10.0 .* cos.(angles); oy = 10.0 .* sin.(angles)
#       for iter in 1:n_iter
#           lr_i = lr * (1.0 - 0.5 * (iter - 1) / n_iter)
#           for i in 1:n
#               lx = pts[i][1] + ox[i]; ly = pts[i][2] + oy[i]
#               fx, fy = 0.0, 0.0
#               for j in 1:n   # repel from other label centres
#                   i == j && continue
#                   dx = lx - (pts[j][1] + ox[j]); dy = ly - (pts[j][2] + oy[j])
#                   d  = sqrt(dx^2 + dy^2) + 0.1
#                   d >= repel && continue
#                   f = ((repel - d) / repel)^2 * repel
#                   fx += f * dx / d; fy += f * dy / d
#               end
#               for j in 1:n   # repel from data points (own point: 3x stronger)
#                   dx = lx - pts[j][1]; dy = ly - pts[j][2]
#                   d  = sqrt(dx^2 + dy^2) + 0.1
#                   d >= repel && continue
#                   s = (i == j) ? 3.0 : 1.0
#                   f = s * ((repel - d) / repel)^2 * repel
#                   fx += f * dx / d; fy += f * dy / d
#               end
#               for (ox_o, oy_o) in obstacles   # repel from fixed obstacles
#                   dx = lx - ox_o; dy = ly - oy_o
#                   d  = sqrt(dx^2 + dy^2) + 0.1
#                   d >= obs_repel && continue
#                   f = ((obs_repel - d) / obs_repel)^2 * obs_repel
#                   fx += f * dx / d; fy += f * dy / d
#               end
#               dx_s = pts[i][1] - lx; dy_s = pts[i][2] - ly   # spring back to own point
#               d_s  = sqrt(dx_s^2 + dy_s^2) + 0.1
#               excess = max(0.0, d_s - dead_zone)
#               f_s  = spring_k * excess^2
#               fx  += f_s * dx_s / d_s; fy += f_s * dy_s / d_s
#               lw, lh = hlw[i], hlh[i]; k = 20.0   # boundary walls
#               lx < lw         && (fx += k * (lw - lx)^1.5)
#               lx > fig_w - lw && (fx -= k * (lx - (fig_w - lw))^1.5)
#               ly < lh         && (fy += k * (lh - ly)^1.5)
#               ly > fig_h - lh && (fy -= k * (ly - (fig_h - lh))^1.5)
#               new_ax = clamp(lx + lr_i * fx, hlw[i], fig_w - hlw[i])
#               new_ay = clamp(ly + lr_i * fy, hlh[i], fig_h - hlh[i])
#               ox[i]  = new_ax - pts[i][1]; oy[i] = new_ay - pts[i][2]
#           end
#       end
#       round.(Int, ox), round.(Int, oy)
#   end
# Call was: _lox_f, _loy_f = _auto_label_offsets_f(PCLASS_DATA_F; obstacles = _QUAD_OBSTACLES_F)
# (with _QUAD_OBSTACLES_F as bare (x,y) points, no radius)

## Screen-space positions of the quadrant corner annotations, so labels avoid them too.
_QUAD_ANCHORS_F = [
    (0.015, 0.63,  :left,  :bottom),
    (0.985, 0.63,  :right, :bottom),
    (0.015, 44.0,  :left,  :top),
    (0.985, 44.0,  :right, :top),
]
_quad_d2s_f(h, m; yrange = (log10(0.60), log10(50.0)), fig_w = 620.0, fig_h = 380.0) =
    ((h - 0.0) / 1.0 * fig_w, (log10(m) - yrange[1]) / (yrange[2] - yrange[1]) * fig_h)
_QUAD_OBSTACLES_F = [
    (px, py, 42.0)   # (x, y, radius) — approximate footprint of the 2-line italic annotation
    for (hc, mc, ha, va) in _QUAD_ANCHORS_F
    for (px, py) in (_quad_d2s_f(hc, mc),)
]

_lox_f, _loy_f = auto_label_offsets(PCLASS_DATA_F;
    xrange = (0.0, 1.0), yrange = (log10(0.60), log10(50.0)), fig_w = 620.0, fig_h = 380.0,
    fontsize_of = it -> it[5] == :bip ? 9.0 : 8.0,
    bold_of     = it -> it[5] == :bip,
    obstacles   = _QUAD_OBSTACLES_F,
)

## ── Figure ───────────────────────────────────────────────────────────────────

mkpath("plots")

@assert all(m <= 50 for (_, _, m, _, _, _) in PCLASS_DATA_F) "A problem exceeds the y-axis limit of 50 modes"

fig_f = Figure(; size = (800, 460))

ax_f = Axis(fig_f[1, 1];
    xlabel      = "Normalised posterior entropy",
    ylabel      = "Mode count",
    yscale      = log10,
    xticks      = 0.0:0.1:1.0,
    yticks      = ([1, 2, 5, 10, 20, 50], ["1","2","5","10","20","50"]),
    xgridcolor       = (:black, 0.08),
    ygridcolor       = (:black, 0.08),
    backgroundcolor  = :white,
)
xlims!(ax_f, 0.0, 1.0)
ylims!(ax_f, 0.60, 50)

## Threshold lines
vlines!(ax_f, [0.50]; color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)
hlines!(ax_f, [1.5];  color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)

## Quadrant annotations — corners
for (hc, ha, mc, va, lbl) in [
        (0.015, :left,  0.63, :bottom, "compact\nunimodal"),
        (0.985, :right, 0.63, :bottom, "diffuse\nunimodal"),
        (0.015, :left,  44.0, :top,    "compact\nmultimodal"),
        (0.985, :right, 44.0, :top,    "diffuse\nmultimodal"),
    ]
    text!(ax_f, hc, mc; text=lbl, color=(RGBf(0.05, 0.20, 0.05), 0.5), fontsize=10,
          align=(ha, va), font=:italic)
end

## Scatter + labels
## Marker fill: HD (5D) problems → filled; 2D problems → hollow (white fill,
## colored outline). Independent of grp (which now only drives label case/weight).
for (i, (lbl, h, m, d, grp, is_hd)) in enumerate(PCLASS_DATA_F)
    col = margin_color_f(d)
    mf  = Float64(m)

    if is_hd
        scatter!(ax_f, [h], [mf];
            color       = col,
            markersize  = 11,
            strokewidth = 0.8,
            strokecolor = (:black, 0.35),
        )
    else
        scatter!(ax_f, [h], [mf];
            color       = (:white, 1.0),
            markersize  = 7,
            strokewidth = 1.5,
            strokecolor = col,
        )
    end

    text!(ax_f, h, mf;
        text     = lbl,
        color    = col,
        fontsize = grp == :bip ? 9.0 : 8.0,
        font     = grp == :bip ? :bold : :regular,
        align    = (:center, :center),
        offset   = Point2f(_lox_f[i], _loy_f[i]),
    )
end

## Colorbar — resample margin_color_f itself (not the raw 3-color _CMAP_F) so the
## displayed bands actually show the discrete step scheme, matching what the
## scatter points render.
_cmap_display_f = let n = 256
    cgrad([margin_color_f(-D_MAX_F + 2*D_MAX_F*(i-1)/(n-1)) for i in 1:n])
end
Colorbar(fig_f[1, 2];
    colormap   = _cmap_display_f,
    limits     = (-D_MAX_F, D_MAX_F),
    ticks      = ([-D_MAX_F, -D_SAT_F, -D_SMALL_F, 0, D_SMALL_F, D_SAT_F, D_MAX_F],
                  ["-1\n(EIV wins)", "-$(D_SAT_F)", "-$(D_SMALL_F)", "0", "$(D_SMALL_F)", "$(D_SAT_F)", "+1\n(MaxVar wins)"]),
    label      = "Score margin: EIV − MaxVar  (unpaired median-curve AUC diff, shared window, range [-1,1])",
    ticklabelsize = 11,
    labelsize  = 12,
    labelpadding = -50,
    width      = 14,
    tellheight = false,
)

colsize!(fig_f.layout, 1, Relative(0.85))

dq_add_banner!(fig_f, DQ_NOTES_F; max_shown=6, fontsize=10)

## Save
save("plots/posterior_classification_final.png", fig_f; px_per_unit = 3)
save("plots/posterior_classification_final.pdf", fig_f)
@info "Saved → plots/posterior_classification_final.{png,pdf}"
