"""
Smoothness classification scatter plot — two panels, one figure:
  (left)  Standard vs. nonstationary GP (nongp)
  (right) Standard vs. WarpedGP

Both panels share the same axes: x = cv_value (proxy value CV over the domain,
targets the failure mode WarpedGP fixes), y = cv_grad (proxy gradient-norm CV,
targets the failure mode nonstationary GP fixes). Point color encodes the score
margin (advanced surrogate − standard, median final log-TV): negative/cool =
advanced surrogate wins, positive/warm = standard wins.

nongp only ran on the 7 original BIP problems (no Group B/C/D data), so the left
panel has far fewer points than the right (37).

Data is read dynamically from:
  plots/classify_smoothness.csv   — cv_value, cv_grad per problem
  plots/nongp_scores_fair.csv     — standard_median, nongp_median (7 problems)
  plots/warpedgp_scores_fair.csv  — standard_median, warpedgp_median (37 problems)

The "_fair" score CSVs (from compute_fair_scores.jl) truncate BOTH configs
being compared to the SAME shared iteration budget per problem, unlike the
original "_final" CSVs which score each run at its own last valid iteration
(unfair whenever the two configs ran different iteration counts, e.g. WarpedGP
got 201 iters vs Standard's ~103 on the 5D BIP problems). Several problems'
fair comparisons are therefore based on fewer iterations than the paper's
target — flagged directly on the figure.

Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session
(after include("src/compute_acq_scores_final.jl"); include("src/compute_fair_scores.jl")).
Results saved to plots/smoothness_classification.{png,pdf}.
"""

using CairoMakie

## ── Color scale (shared by both panels) ──────────────────────────────────────

const D_MAX_SM       = 1.0
const DRAW_THRESH_SM = log(1.20)   # ≈ 0.182 — 20% better to count as a win

function margin_color_sm(diff::Real)
    d = clamp(Float64(diff), -D_MAX_SM, D_MAX_SM)
    t = if d <= -DRAW_THRESH_SM
            0.25 * (d + D_MAX_SM) / (D_MAX_SM - DRAW_THRESH_SM)
        elseif d <= DRAW_THRESH_SM
            0.5 + 0.1 * d / DRAW_THRESH_SM
        else
            0.75 + 0.25 * (d - DRAW_THRESH_SM) / (D_MAX_SM - DRAW_THRESH_SM)
        end
    c = cgrad(:coolwarm)[t]
    f = 0.80f0
    return RGBf(c.r*f, c.g*f, c.b*f)
end

## ── Load data ────────────────────────────────────────────────────────────────

function _read_csv_cols_sm(path, cols)
    lines = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

_cls_rows_sm = _read_csv_cols_sm("plots/classify_smoothness.csv",
                                  ["problem", "cv_value", "cv_grad"])

_nongp_rows_sm = _read_csv_cols_sm("plots/nongp_scores_fair.csv",
                                     ["problem", "nongp_median", "standard_median"])
_nongp_dict_sm = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in _nongp_rows_sm)

_wgp_rows_sm = _read_csv_cols_sm("plots/warpedgp_scores_fair.csv",
                                   ["problem", "warpedgp_median", "standard_median"])
_wgp_dict_sm = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in _wgp_rows_sm)

const _BIP_NAMES_SM = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                            "SIRProblem", "DuffingProblem", "DiffusionProblem10",
                            "DuffingProblem5", "DiffusionProblem5D"])

function _display_label_sm(prob)
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

# (label, cv_value, cv_grad, diff, group) — one list per panel, restricted to
# problems that actually have score data for that surrogate.
function _panel_data_sm(score_dict)
    [(_display_label_sm(r[1]), parse(Float64, r[2]), parse(Float64, r[3]),
      score_dict[r[1]], r[1] in _BIP_NAMES_SM ? :bip : :opt)
     for r in _cls_rows_sm if haskey(score_dict, r[1])]
end

const NONGP_DATA_SM = _panel_data_sm(_nongp_dict_sm)
const WGP_DATA_SM    = _panel_data_sm(_wgp_dict_sm)

## ── Automatic label placement via force-directed repulsion (linear axes) ─────

function _auto_label_offsets_sm(items;
    xrange    = (0.0, 1.15),
    yrange    = (0.0, 1.10),
    fig_w     = 380.0,
    fig_h     = 420.0,
    n_iter    = 300,
    lr        = 0.25,
    repel     = 45.0,
    spring_k  = 0.012,
    dead_zone = 10.0)

    n      = length(items)
    labels = [x[1] for x in items]
    is_bip = [x[5] == :bip for x in items]
    fs     = [b ? 9.0 : 8.0 for b in is_bip]
    hlw    = [length(labels[i]) * fs[i] * 0.55 / 2 for i in 1:n]
    hlh    = [fs[i] * 0.6 for i in 1:n]

    function d2s(x, y)
        px = (x - xrange[1]) / (xrange[2] - xrange[1]) * fig_w
        py = (y - yrange[1]) / (yrange[2] - yrange[1]) * fig_h
        (px, py)
    end
    pts = [d2s(it[2], it[3]) for it in items]

    angles = LinRange(0.0, 2π - 2π/max(n,1), n)
    ox = 10.0 .* cos.(angles)
    oy = 10.0 .* sin.(angles)

    for iter in 1:n_iter
        lr_i = lr * (1.0 - 0.5 * (iter - 1) / n_iter)
        for i in 1:n
            lx = pts[i][1] + ox[i]
            ly = pts[i][2] + oy[i]
            fx, fy = 0.0, 0.0

            for j in 1:n
                i == j && continue
                dx = lx - (pts[j][1] + ox[j])
                dy = ly - (pts[j][2] + oy[j])
                d  = sqrt(dx^2 + dy^2) + 0.1
                d >= repel && continue
                f = ((repel - d) / repel)^2 * repel
                fx += f * dx / d
                fy += f * dy / d
            end

            for j in 1:n
                dx = lx - pts[j][1]
                dy = ly - pts[j][2]
                d  = sqrt(dx^2 + dy^2) + 0.1
                d >= repel && continue
                s = (i == j) ? 3.0 : 1.0
                f = s * ((repel - d) / repel)^2 * repel
                fx += f * dx / d
                fy += f * dy / d
            end

            dx_s  = pts[i][1] - lx
            dy_s  = pts[i][2] - ly
            d_s   = sqrt(dx_s^2 + dy_s^2) + 0.1
            excess = max(0.0, d_s - dead_zone)
            f_s   = spring_k * excess^2
            fx   += f_s * dx_s / d_s
            fy   += f_s * dy_s / d_s

            lw, lh = hlw[i], hlh[i]
            k = 20.0
            lx < lw           && (fx += k * (lw - lx)^1.5)
            lx > fig_w - lw   && (fx -= k * (lx - (fig_w - lw))^1.5)
            ly < lh           && (fy += k * (lh - ly)^1.5)
            ly > fig_h - lh   && (fy -= k * (ly - (fig_h - lh))^1.5)

            new_ax = clamp(lx + lr_i * fx, hlw[i], fig_w - hlw[i])
            new_ay = clamp(ly + lr_i * fy, hlh[i], fig_h - hlh[i])
            ox[i]  = new_ax - pts[i][1]
            oy[i]  = new_ay - pts[i][2]
        end
    end

    round.(Int, ox), round.(Int, oy)
end

_lox_nongp_sm, _loy_nongp_sm = _auto_label_offsets_sm(NONGP_DATA_SM)
_lox_wgp_sm,   _loy_wgp_sm   = _auto_label_offsets_sm(WGP_DATA_SM)

## ── Panel drawing helper ──────────────────────────────────────────────────────

function _draw_panel_sm!(fig, col, data, lox, loy, title)
    ax = Axis(fig[1, col];
        xlabel = "cv_value  (proxy value CV)",
        ylabel = col == 1 ? "cv_grad  (proxy gradient-norm CV)" : "",
        title  = title,
        xticks = 0.0:0.2:1.0,
        yticks = 0.0:0.2:1.0,
        xgridcolor      = (:black, 0.08),
        ygridcolor      = (:black, 0.08),
        backgroundcolor = :white,
    )
    xlims!(ax, 0.0, 1.15)
    ylims!(ax, 0.0, 1.10)

    for (i, (lbl, cvv, cvg, d, grp)) in enumerate(data)
        col_ = margin_color_sm(d)
        if grp == :bip
            scatter!(ax, [cvv], [cvg];
                color = col_, markersize = 11, strokewidth = 0.8, strokecolor = (:black, 0.35))
        else
            scatter!(ax, [cvv], [cvg];
                color = (:white, 1.0), markersize = 7, strokewidth = 1.5, strokecolor = col_)
        end
        text!(ax, cvv, cvg;
            text = lbl, color = col_,
            fontsize = grp == :bip ? 9.0 : 8.0,
            font     = grp == :bip ? :bold : :regular,
            align    = (:center, :center),
            offset   = Point2f(lox[i], loy[i]),
        )
    end
    return ax
end

## ── Figure ───────────────────────────────────────────────────────────────────

mkpath("plots")

fig_sm = Figure(; size = (1400, 500))

Label(fig_sm[0, 1:4], "TODO: calculated only with available iters — comparisons below use a shared per-problem iteration budget (min across both configs), not each config's full run";
      color = :red, fontsize = 16, font = :bold, padding = (0,0,0,4))

_draw_panel_sm!(fig_sm, 1, NONGP_DATA_SM, _lox_nongp_sm, _loy_nongp_sm,
                "Standard GP vs. nonstationary GP  (n=$(length(NONGP_DATA_SM)) problems)")

Colorbar(fig_sm[1, 2];
    colormap   = cgrad([margin_color_sm(-D_MAX_SM + 2*D_MAX_SM*(i-1)/255) for i in 1:256]),
    limits     = (-D_MAX_SM, D_MAX_SM),
    ticks      = ([-D_MAX_SM, -DRAW_THRESH_SM, 0, DRAW_THRESH_SM, D_MAX_SM],
                  ["< -1\n(NonstatGP wins)", "-0.2", "0", "+0.2", "> +1\n(Standard wins)"]),
    label      = "Score margin: NonstatGP − Standard",
    ticklabelsize = 10, labelsize = 11, labelpadding = -50, width = 14, tellheight = false,
)

_draw_panel_sm!(fig_sm, 3, WGP_DATA_SM, _lox_wgp_sm, _loy_wgp_sm,
                "Standard GP vs. WarpedGP  (n=$(length(WGP_DATA_SM)) problems)")

Colorbar(fig_sm[1, 4];
    colormap   = cgrad([margin_color_sm(-D_MAX_SM + 2*D_MAX_SM*(i-1)/255) for i in 1:256]),
    limits     = (-D_MAX_SM, D_MAX_SM),
    ticks      = ([-D_MAX_SM, -DRAW_THRESH_SM, 0, DRAW_THRESH_SM, D_MAX_SM],
                  ["< -1\n(WarpedGP wins)", "-0.2", "0", "+0.2", "> +1\n(Standard wins)"]),
    label      = "Score margin: WarpedGP − Standard",
    ticklabelsize = 10, labelsize = 11, labelpadding = -50, width = 14, tellheight = false,
)

colsize!(fig_sm.layout, 1, Relative(0.40))
colsize!(fig_sm.layout, 3, Relative(0.40))

save("plots/smoothness_classification.png", fig_sm; px_per_unit = 3)
save("plots/smoothness_classification.pdf", fig_sm)
@info "Saved → plots/smoothness_classification.{png,pdf}"
