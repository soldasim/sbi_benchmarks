"""
Non-homogeneity criterion H's homogeneity p-value vs WarpedGP performance margin.

Tests paper1_suggestions.md Suggestion 1's prediction P1 directly: "large
warped-GP gains occur if and only if the homogeneous null is rejected" — i.e.
a scatter of the homogeneity p-value (x) against the actual measured
WarpedGP-vs-Standard performance margin (y) should show WarpedGP-winning
margins concentrated at low p.

Naming note: this is the HOMOGENEITY p-value, not a "non-homogeneity p-value"
— by convention a p-value is named for its null hypothesis, and the null being
tested here is that the problem IS homogeneous (see classify_nonhomogeneity.jl).
Small p = reject homogeneity = evidence of non-homogeneity; large p = failed to
reject homogeneity (NOT evidence FOR homogeneity — could just mean the test was
underpowered for that problem). "Non-homogeneity p-value" would misleadingly
read as if non-homogeneity were the null being tested.

## Axes

  x = p  — homogeneity-null bootstrap p-value from classify_nonhomogeneity.jl
           (worst-dimension aggregate already stored in the CSV). Plotted on a
           LINEAR scale, not log: several problems have p exactly 0.0 (9 of 37),
           which a log axis can't represent without an arbitrary floor, and at
           B=200 there are only ~11 distinct values below the conventional 0.05
           threshold anyway, so linear loses no real resolution there.
  y = -warpedgp_diff — the NEGATED unpaired, difference-of-median-curves AUC
           score from compute_median_curve_auc_scores.jl (raw column has
           candidate=WarpedGP, baseline=Standard, negative=WarpedGP wins; negated
           here so POSITIVE = WarpedGP wins, matching the intuitive "up = good
           for the thing we're testing" reading). Theoretical range is [-1,1],
           but the actual data only spans roughly [-0.08, 0.35], so the y-axis
           is auto-fit to the data range (with padding) rather than fixed to
           [-1,1] — fixing it would compress every point into a thin band near 0.

A vertical dashed line marks p=0.05 (conventional significance threshold); a
horizontal dashed line marks the margin=0 (draw).

Point color is deliberately NOT used to encode win/lose — the y-position
already shows that directly, and a redundant color channel just repeats the
same information as a second (harder-to-read) encoding. Color/marker-fill is
reserved for problem group (BIP-family bold, HD-problem filled marker), same
as the other classification scatter plots in this repo.

## Correlation

Spearman rank correlation (not Pearson — p is bounded/skewed, not linearly
related to anything) between p and the (negated) margin, reported with its own
two-sided significance (t-approximation, df=n-2), printed to stdout and
annotated on the figure. Rank-based sidesteps the p=0 ties issue entirely (no
log transform needed anywhere in this file). Sign of ρ flips relative to the
raw (non-negated) column, since it's computed on the negated/plotted variable.

## Scope

All 37 problems — classify_nonhomogeneity.csv and warpedgp_scores_auc.csv have
been verified to cover the exact same 37 problem names (no join mismatches).

Does NOT test P2 (SIR raw vs log-proxy shrinking H) — that needs H computed for
the proxy-variant problems, which the original 37-problem H batch didn't cover
(non-proxy-everywhere convention). Separate follow-up if wanted.

Data is read dynamically from:
  plots/classify_nonhomogeneity.csv   — problem, dx, p
  plots/warpedgp_scores_auc.csv       — problem, warpedgp_diff

Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.
Results saved to plots/nonhomogeneity_vs_warpedgp.{png,pdf}.
"""

using CairoMakie
using Statistics: mean
using Distributions: TDist, ccdf

include("label_placement.jl")

## ── Load data ────────────────────────────────────────────────────────────────

function _read_csv_cols_nh(path, cols)
    lines = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

_h_rows   = _read_csv_cols_nh("plots/classify_nonhomogeneity.csv", ["problem", "dx", "p"])
_h_dict   = Dict(r[1] => (dx=parse(Int, r[2]), p=parse(Float64, r[3])) for r in _h_rows)

_wgp_rows = _read_csv_cols_nh("plots/warpedgp_scores_auc.csv", ["problem", "warpedgp_diff"])
_wgp_dict = Dict(r[1] => parse(Float64, r[2]) for r in _wgp_rows)

## :bip = the 9 "BIP-family" problems (7 orig + 2 HD BIP) — drives marker
## bold/size only, matching the other classification scatter plots.
const _BIP_NAMES_NH = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                            "SIRProblem", "DuffingProblem", "DiffusionProblem10",
                            "DuffingProblem5", "DiffusionProblem5D"])
const _ORIG7_NAMES_NH = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                              "SIRProblem", "DuffingProblem", "DiffusionProblem10"])
const _HD_PROBLEMS_NH = Set(["DuffingProblem5", "DiffusionProblem5D", "RosenbrockProblem5",
                              "StyblinskiTangProblem5", "MichalewiczProblem5", "SphereProblem5"])

function _display_label_nh(prob)
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

## Tuple: (label_text, p, wgp_margin [= -warpedgp_diff, positive = WarpedGP wins], grp, is_hd)
const NHWG_DATA = [
    let grp = prob in _BIP_NAMES_NH ? :bip : :opt,
        raw_lbl = _display_label_nh(prob)
        (prob in _ORIG7_NAMES_NH ? uppercase(raw_lbl) : raw_lbl,
         _h_dict[prob].p,
         -_wgp_dict[prob],
         grp,
         prob in _HD_PROBLEMS_NH)
    end
    for prob in intersect(keys(_h_dict), keys(_wgp_dict))
]
@assert length(NHWG_DATA) == 37 "Expected 37 joined problems, got $(length(NHWG_DATA))"

## ── Spearman correlation ────────────────────────────────────────────────────

function _rank(v::AbstractVector{<:Real})
    n = length(v)
    idx = sortperm(v)
    ranks = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[idx[j+1]] == v[idx[i]]
            j += 1
        end
        r = (i + j) / 2
        for k in i:j
            ranks[idx[k]] = r
        end
        i = j + 1
    end
    return ranks
end

function spearman(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    rx, ry = _rank(x), _rank(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((rx .- mx) .* (ry .- my))
    sx  = sqrt(sum((rx .- mx) .^ 2))
    sy  = sqrt(sum((ry .- my) .^ 2))
    return cov / (sx * sy)
end

_p_vals   = [d[2] for d in NHWG_DATA]
_wgp_vals = [d[3] for d in NHWG_DATA]   # = -warpedgp_diff; positive = WarpedGP wins
_n_nh     = length(NHWG_DATA)
_rho_nh   = spearman(_p_vals, _wgp_vals)
_t_nh     = _rho_nh * sqrt((_n_nh - 2) / (1 - _rho_nh^2))
_pval_nh  = 2 * ccdf(TDist(_n_nh - 2), abs(_t_nh))

println("Spearman ρ(p, -warpedgp_diff) = $(round(_rho_nh; digits=3))  " *
        "(n=$_n_nh, t=$(round(_t_nh; digits=3)), two-sided p=$(round(_pval_nh; digits=4)))")
_rho_nh < 0 && println("Negative ρ: as expected under P1, HIGHER p (less non-homogeneous) associates with a SMALLER WarpedGP-win margin — i.e. low p (significant) should trend toward a larger positive margin (WarpedGP wins).")

## ── Figure ───────────────────────────────────────────────────────────────────

mkpath("plots")

_x_lo, _x_hi = -0.03, 1.0
_y_lo = minimum(_wgp_vals) - 0.03
_y_hi = maximum(_wgp_vals) + 0.03

fig_nh = Figure(; size = (800, 460))

ax_nh = Axis(fig_nh[1, 1];
    xlabel = "Homogeneity p-value (H0: problem is homogeneous)",
    ylabel = "WarpedGP win margin (−warpedgp_diff, unpaired median-curve AUC)",
    xticks = 0.0:0.1:1.0,
    xgridcolor = (:black, 0.08),
    ygridcolor = (:black, 0.08),
    backgroundcolor = :white,
)
xlims!(ax_nh, _x_lo, _x_hi)
ylims!(ax_nh, _y_lo, _y_hi)

## Point color — same discrete 5-step scheme (thresholds 0.01/0.1, same
## coolwarm-derived palette) as margin_color_f in plot_posterior_classification_final.jl,
## applied back to the points/labels (not a background) per request. As before:
## NEGATIVE y (Standard-wins side, since y = -warpedgp_diff here) gets the blue
## shades, POSITIVE y (WarpedGP-wins side) gets red — i.e. blue/red track the
## SIGN of the axis value, the mirror image of the other plot's raw-column
## "blue = candidate/WarpedGP wins" rule (flag if you want that swapped).
_coolwarm_dark_nh(t) = (c = cgrad(:coolwarm)[t]; f = 0.80f0; RGBf(c.r*f, c.g*f, c.b*f))
const _BLUE_NH  = _coolwarm_dark_nh(0.0)
const _LBLUE_NH = _coolwarm_dark_nh(0.25)
const _GREY_NH  = _coolwarm_dark_nh(0.5)
const _LRED_NH  = _coolwarm_dark_nh(0.75)
const _RED_NH   = _coolwarm_dark_nh(1.0)
const _D_SMALL_NH = 0.01
const _D_SAT_NH   = 0.1

function margin_color_nh(y::Real)
    ay = abs(y)
    ay < _D_SMALL_NH && return _GREY_NH
    ay < _D_SAT_NH   && return y > 0 ? _LRED_NH : _LBLUE_NH
    return y > 0 ? _RED_NH : _BLUE_NH
end

hlines!(ax_nh, [0.0]; color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)
vlines!(ax_nh, [0.05]; color = (:firebrick, 0.5), linestyle = :dash, linewidth = 1.0)
text!(ax_nh, 0.05, _y_hi; text = "p = 0.05", color = (:firebrick, 0.7), fontsize = 10,
      align = (:left, :top), offset = Point2f(4, -2))

## y-axis direction annotations, colored to match their region (positive=red=
## WarpedGP wins, negative=blue=Standard wins here — see margin_color_nh above).
## Offset down from y_hi (not flush with it) to avoid colliding with the
## Spearman-ρ annotation, which is also anchored at (x_hi, y_hi).
text!(ax_nh, _x_hi, _y_hi; text = "Warped GP Wins", color = (_RED_NH, 0.7),
      fontsize = 11, font = :italic, align = (:right, :top), offset = Point2f(0, -18))
text!(ax_nh, _x_hi, _y_lo; text = "Standard GP Wins", color = (_BLUE_NH, 0.7),
      fontsize = 11, font = :italic, align = (:right, :bottom))

_lox_nh, _loy_nh = auto_label_offsets(NHWG_DATA;
    xrange = (_x_lo, _x_hi), yrange = (_y_lo, _y_hi), fig_w = 620.0, fig_h = 380.0,
    fontsize_of = it -> it[4] == :bip ? 9.0 : 8.0,
    bold_of     = it -> it[4] == :bip,
    ylog        = false,
)

for (i, (lbl, p, margin, grp, is_hd)) in enumerate(NHWG_DATA)
    col = margin_color_nh(margin)
    if is_hd
        scatter!(ax_nh, [p], [margin]; color = col, markersize = 11, strokewidth = 0.8, strokecolor = (:black, 0.35))
    else
        scatter!(ax_nh, [p], [margin]; color = (:white, 1.0), markersize = 7, strokewidth = 1.5, strokecolor = col)
    end
    text!(ax_nh, p, margin;
        text = lbl, color = col,
        fontsize = grp == :bip ? 9.0 : 8.0,
        font = grp == :bip ? :bold : :regular,
        align = (:center, :center),
        offset = Point2f(_lox_nh[i], _loy_nh[i]),
    )
end

text!(ax_nh, _x_hi, _y_hi;
    text = "Spearman ρ = $(round(_rho_nh; digits=3))  (n=$_n_nh, p=$(round(_pval_nh; digits=4)))",
    color = (:black, 0.7), fontsize = 11, align = (:right, :top),
)

save("plots/nonhomogeneity_vs_warpedgp.png", fig_nh; px_per_unit = 3)
save("plots/nonhomogeneity_vs_warpedgp.pdf", fig_nh)
@info "Saved → plots/nonhomogeneity_vs_warpedgp.{png,pdf}"
