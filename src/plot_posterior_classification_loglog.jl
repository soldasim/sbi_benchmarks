"""
Posterior classification scatter plot — log-log area score variant.

Same layout as plot_posterior_classification.jl, but the colour axis uses
the log-log area score (area under log(TV) vs log(iteration), normalised)
instead of the uniform mean-log-TV.

Data is read dynamically from:
  plots/classify_posteriors.csv   — h_rel, n_modes per problem
  plots/acq_scores_loglog.csv     — eiv_median, maxvar_median per problem

Run from ~/repos/bosip_benchmarks/ via include() in a live Julia session.
Results saved to plots/posterior_classification_loglog.{png,pdf}.
"""

using CairoMakie

## ── Color scale ──────────────────────────────────────────────────────────────

const D_MAX_LL       = 1.0
const DRAW_THRESH_LL = log(1.20)   # ≈ 0.182 — 20% better to count as a win

function margin_color_ll(diff::Real)
    d = clamp(Float64(diff), -D_MAX_LL, D_MAX_LL)
    t = if d <= -DRAW_THRESH_LL
            0.25 * (d + D_MAX_LL) / (D_MAX_LL - DRAW_THRESH_LL)
        elseif d <= DRAW_THRESH_LL
            0.5 + 0.1 * d / DRAW_THRESH_LL
        else
            0.75 + 0.25 * (d - DRAW_THRESH_LL) / (D_MAX_LL - DRAW_THRESH_LL)
        end
    c = cgrad(:coolwarm)[t]
    f = 0.80f0
    return RGBf(c.r*f, c.g*f, c.b*f)
end


## ── Load data ────────────────────────────────────────────────────────────────

function _read_csv_cols(path, cols)
    lines = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    [Tuple(split(l, ",")[i] for i in idxs) for l in lines[2:end] if !isempty(strip(l))]
end

# scores: problem → diff (eiv_median − maxvar_median)
_scr_rows  = _read_csv_cols("plots/acq_scores_loglog.csv",
                             ["problem", "eiv_median", "maxvar_median"])
_scr_dict  = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in _scr_rows)

# classification: problem → (H_rel, n_modes)
_cls_rows  = _read_csv_cols("plots/classify_posteriors.csv",
                             ["problem", "H_rel", "n_modes"])

const _BIP_NAMES = Set(["ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
                         "SIRProblem", "DuffingProblem", "DiffusionProblem10"])

function _display_label(prob)
    lbl = replace(prob, r"Problem\d*$" => "")
    lbl = replace(lbl, "ExpandedSchafferF6" => "SchafferF6")
    lbl = replace(lbl, "ExpandedZakharov"   => "Zakharov")
    lbl = replace(lbl, "StyblinskiTang"     => "StybTang")
    lbl = replace(lbl, "GoldsteinPrice"     => "Goldstein")
    lbl = replace(lbl, "ThreeHumpCamel"     => "3HumpCamel")
    return lbl
end

const PCLASS_DATA_LL = [
    (_display_label(r[1]),
     parse(Float64, r[2]),
     parse(Int,     r[3]),
     _scr_dict[r[1]],
     r[1] in _BIP_NAMES ? :bip : :opt)
    for r in _cls_rows if haskey(_scr_dict, r[1])
]

## ── Automatic label placement via force-directed repulsion ───────────────────

function _auto_label_offsets_ll(items;
    xrange    = (0.0, 1.0),
    yrange    = (log10(0.60), log10(50.0)),
    fig_w     = 620.0,
    fig_h     = 380.0,
    n_iter    = 300,
    lr        = 0.25,
    repel     = 50.0,
    spring_k  = 0.012,
    dead_zone = 10.0,
    obstacles = Tuple{Float64,Float64}[])

    n      = length(items)
    labels = [x[1] for x in items]
    is_bip = [x[5] == :bip for x in items]
    fs     = [b ? 9.0 : 8.0 for b in is_bip]
    hlw    = [length(labels[i]) * fs[i] * 0.55 / 2 for i in 1:n]
    hlh    = [fs[i] * 0.6 for i in 1:n]

    function d2s(h, m)
        x = (h - xrange[1]) / (xrange[2] - xrange[1]) * fig_w
        y = (log10(Float64(m)) - yrange[1]) / (yrange[2] - yrange[1]) * fig_h
        (x, y)
    end
    pts = [d2s(x[2], x[3]) for x in items]
    obs_repel = 60.0

    angles = LinRange(0.0, 2π - 2π/n, n)
    ox = 10.0 .* cos.(angles)
    oy = 10.0 .* sin.(angles)

    for iter in 1:n_iter
        lr_i = lr * (1.0 - 0.5 * (iter - 1) / n_iter)
        for i in 1:n
            lx = pts[i][1] + ox[i]
            ly = pts[i][2] + oy[i]
            fx, fy = 0.0, 0.0

            # Repel from other label centres
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

            # Repel from data points (own point: 3× stronger)
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

            # Repel from fixed obstacles (e.g. quadrant annotations)
            for (ox_o, oy_o) in obstacles
                dx = lx - ox_o
                dy = ly - oy_o
                d  = sqrt(dx^2 + dy^2) + 0.1
                d >= obs_repel && continue
                f = ((obs_repel - d) / obs_repel)^2 * obs_repel
                fx += f * dx / d
                fy += f * dy / d
            end

            # Quadratic spring — keeps label near its own point
            dx_s  = pts[i][1] - lx
            dy_s  = pts[i][2] - ly
            d_s   = sqrt(dx_s^2 + dy_s^2) + 0.1
            excess = max(0.0, d_s - dead_zone)
            f_s   = spring_k * excess^2
            fx   += f_s * dx_s / d_s
            fy   += f_s * dy_s / d_s

            # Strong boundary walls
            lw, lh = hlw[i], hlh[i]
            k = 20.0
            lx < lw           && (fx += k * (lw - lx)^1.5)
            lx > fig_w - lw   && (fx -= k * (lx - (fig_w - lw))^1.5)
            ly < lh           && (fy += k * (lh - ly)^1.5)
            ly > fig_h - lh   && (fy -= k * (ly - (fig_h - lh))^1.5)

            # Clamp absolute label position during iteration
            new_ax = clamp(lx + lr_i * fx, hlw[i], fig_w - hlw[i])
            new_ay = clamp(ly + lr_i * fy, hlh[i], fig_h - hlh[i])
            ox[i]  = new_ax - pts[i][1]
            oy[i]  = new_ay - pts[i][2]
        end
    end

    round.(Int, ox), round.(Int, oy)
end

## Screen-space positions of the quadrant corner annotations, so labels avoid them too
_QUAD_ANCHORS_LL = [
    (0.015, 0.63,  :left,  :bottom),
    (0.985, 0.63,  :right, :bottom),
    (0.015, 44.0,  :left,  :top),
    (0.985, 44.0,  :right, :top),
]
_quad_d2s_ll(h, m; yrange = (log10(0.60), log10(50.0)), fig_w = 620.0, fig_h = 380.0) =
    ((h - 0.0) / 1.0 * fig_w, (log10(m) - yrange[1]) / (yrange[2] - yrange[1]) * fig_h)
_QUAD_OBSTACLES_LL = [
    (px + (ha == :left ? 25.0 : -25.0), py + (va == :bottom ? 12.0 : -12.0))
    for (hc, mc, ha, va) in _QUAD_ANCHORS_LL
    for (px, py) in (_quad_d2s_ll(hc, mc),)
]

_lox_ll, _loy_ll = _auto_label_offsets_ll(PCLASS_DATA_LL; obstacles = _QUAD_OBSTACLES_LL)

## ── Figure ───────────────────────────────────────────────────────────────────

mkpath("plots")

@assert all(m <= 50 for (_, _, m, _, _) in PCLASS_DATA_LL) "A problem exceeds the y-axis limit of 50 modes"

fig_ll = Figure(; size = (800, 460))

ax_ll = Axis(fig_ll[1, 1];
    xlabel      = "Normalised posterior entropy",
    ylabel      = "Mode count",
    yscale      = log10,
    xticks      = 0.0:0.1:1.0,
    yticks      = ([1, 2, 5, 10, 20, 50], ["1","2","5","10","20","50"]),
    xgridcolor       = (:black, 0.08),
    ygridcolor       = (:black, 0.08),
    backgroundcolor  = :white,
)
xlims!(ax_ll, 0.0, 1.0)
ylims!(ax_ll, 0.60, 50)

## Threshold lines
vlines!(ax_ll, [0.50]; color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)
hlines!(ax_ll, [1.5];  color = (:black, 0.28), linestyle = :dash, linewidth = 1.0)

## Quadrant annotations — corners
for (hc, ha, mc, va, lbl) in [
        (0.015, :left,  0.63, :bottom, "compact\nunimodal"),
        (0.985, :right, 0.63, :bottom, "diffuse\nunimodal"),
        (0.015, :left,  44.0, :top,    "compact\nmultimodal"),
        (0.985, :right, 44.0, :top,    "diffuse\nmultimodal"),
    ]
    text!(ax_ll, hc, mc; text=lbl, color=(RGBf(0.05, 0.20, 0.05), 0.5), fontsize=10,
          align=(ha, va), font=:italic)
end

## Scatter + labels
for (i, (lbl, h, m, d, grp)) in enumerate(PCLASS_DATA_LL)
    col = margin_color_ll(d)
    mf  = Float64(m)

    if grp == :bip
        scatter!(ax_ll, [h], [mf];
            color       = col,
            markersize  = 11,
            strokewidth = 0.8,
            strokecolor = (:black, 0.35),
        )
    else
        scatter!(ax_ll, [h], [mf];
            color       = (:white, 1.0),
            markersize  = 7,
            strokewidth = 1.5,
            strokecolor = col,
        )
    end

    text!(ax_ll, h, mf;
        text     = lbl,
        color    = col,
        fontsize = grp == :bip ? 9.0 : 8.0,
        font     = grp == :bip ? :bold : :regular,
        align    = (:center, :center),
        offset   = Point2f(_lox_ll[i], _loy_ll[i]),
    )
end

## Colorbar
_cmap_ll = let n = 256
    cgrad([margin_color_ll(-D_MAX_LL + 2*D_MAX_LL*(i-1)/(n-1)) for i in 1:n])
end
Colorbar(fig_ll[1, 2];
    colormap   = _cmap_ll,
    limits     = (-D_MAX_LL, D_MAX_LL),
    ticks      = ([-D_MAX_LL, -DRAW_THRESH_LL, 0, DRAW_THRESH_LL, D_MAX_LL],
                  ["< -$(Int(D_MAX_LL))\n(EIV wins)", "-0.2", "0", "+0.2",
                   "> +$(Int(D_MAX_LL))\n(MaxVar wins)"]),
    label      = "Score margin: EIV − MaxVar  (median log-log AUC)",
    ticklabelsize = 11,
    labelsize  = 12,
    labelpadding = -50,
    width      = 14,
    tellheight = false,
)

colsize!(fig_ll.layout, 1, Relative(0.85))

## Save
save("plots/posterior_classification_loglog.png", fig_ll; px_per_unit = 3)
save("plots/posterior_classification_loglog.pdf", fig_ll)
@info "Saved → plots/posterior_classification_loglog.{png,pdf}"
