# Greedy candidate-position label placement for scatter plots.
#
# Replaces the older force-directed/spring-relaxation placer (which let labels
# drift far from their point when the canvas was crowded — see the 2026-08-06
# fix). For each point, in most-crowded-first order, try a ring of candidate
# offsets at increasing radius and take the CLOSEST one that doesn't overlap
# any already-placed label, any data point, or any fixed obstacle. This is
# greedy (order-dependent, not globally optimal) but keeps labels visibly
# anchored to their point, which is what matters more for a 30-40 point
# scientific scatter plot than a perfect global optimum.
#
# Safe to `include("label_placement.jl")` repeatedly (functions only, no
# structs/consts).

# items: vector of tuples where item[1] = label text (String), item[2] = x data
# coordinate, item[3] = y data coordinate (linear or to be log10'd via yrange).
# fontsize_of/bold_of extract per-item label styling to estimate its box size.
# obstacles: vector of (x, y, radius) in the same screen-space units as fig_w/fig_h.
function auto_label_offsets(items;
    xrange,
    yrange,
    fig_w,
    fig_h,
    fontsize_of  = _ -> 8.0,
    bold_of      = _ -> false,
    obstacles    = Tuple{Float64,Float64,Float64}[],
    point_radius = 4.0,
    radii        = (8.0, 12.0, 17.0, 23.0, 30.0, 38.0, 48.0, 60.0, 75.0, 92.0),
    n_angles     = 16,
    ylog         = true,   # y-axis is log10-scaled (mode count etc); set false for a linear y-axis (can then be negative)
)
    n      = length(items)
    labels = [it[1] for it in items]
    fs     = [fontsize_of(it) for it in items]
    bold   = [bold_of(it) for it in items]
    charw  = [b ? 0.62 : 0.55 for b in bold]
    hlw    = [length(labels[i]) * fs[i] * charw[i] / 2 for i in 1:n]
    hlh    = [fs[i] * 0.65 for i in 1:n]

    function d2s(x, y)
        sx = (x - xrange[1]) / (xrange[2] - xrange[1]) * fig_w
        yv = ylog ? log10(Float64(y)) : Float64(y)
        sy = (yv - yrange[1]) / (yrange[2] - yrange[1]) * fig_h
        (sx, sy)
    end
    pts = [d2s(it[2], it[3]) for it in items]

    # Most-crowded-first: points with more neighbors within 70 units get first
    # pick of the free space around them.
    neighbor_count = [count(j -> j != i && hypot(pts[j][1]-pts[i][1], pts[j][2]-pts[i][2]) < 70, 1:n) for i in 1:n]
    order = sortperm(neighbor_count; rev=true)

    box_overlaps(cx, cy, hw, hh, bx, by, bhw, bhh) =
        abs(cx - bx) < (hw + bhw) && abs(cy - by) < (hh + bhh)

    circle_overlaps_box(cx, cy, hw, hh, px, py, r) = begin
        dx = clamp(px, cx - hw, cx + hw) - px
        dy = clamp(py, cy - hh, cy + hh) - py
        dx*dx + dy*dy < r*r
    end

    placed_boxes = Tuple{Float64,Float64,Float64,Float64}[]   # (cx, cy, halfw, halfh)
    ox = zeros(n); oy = zeros(n)

    for i in order
        px, py = pts[i]
        hw, hh = hlw[i], hlh[i]
        best = nothing
        for r in radii
            found_at_r = nothing
            for k in 0:(n_angles - 1)
                # Half-step offset: avoids testing exact cardinal directions
                # (0/90/180/270°) first — those are exactly where axis
                # gridlines and dashed threshold lines run, so a label (and
                # its leader line back to the point) placed dead-on one of
                # them reads as "part of the gridline" rather than a distinct
                # connector (see the 2026-08-06 "Booth" case).
                ang = 2π * (k + 0.5) / n_angles
                cx = px + r * cos(ang)
                cy = py + r * sin(ang)
                (cx - hw < 0 || cx + hw > fig_w || cy - hh < 0 || cy + hh > fig_h) && continue

                bad = false
                for j in 1:n
                    if circle_overlaps_box(cx, cy, hw, hh, pts[j][1], pts[j][2], point_radius)
                        bad = true; break
                    end
                end
                if !bad
                    for (bx, by, bhw, bhh) in placed_boxes
                        if box_overlaps(cx, cy, hw, hh, bx, by, bhw, bhh)
                            bad = true; break
                        end
                    end
                end
                if !bad
                    for (obx, oby, obr) in obstacles
                        if circle_overlaps_box(cx, cy, hw, hh, obx, oby, obr)
                            bad = true; break
                        end
                    end
                end
                if !bad
                    found_at_r = (cx, cy)
                    break
                end
            end
            if found_at_r !== nothing
                best = found_at_r
                break
            end
        end
        if best === nothing
            # Fallback: nothing worked in the tried radii — place at the
            # largest radius, clamped inside the canvas, accepting overlap.
            r = radii[end]
            cx = clamp(px + r, hw, fig_w - hw)
            cy = clamp(py, hh, fig_h - hh)
            best = (cx, cy)
        end
        ox[i] = best[1] - px
        oy[i] = best[2] - py
        push!(placed_boxes, (best[1], best[2], hw, hh))
    end

    round.(Int, ox), round.(Int, oy)
end

# Inverse of the d2s transform used above: given a data point (x, y) and the
# pixel offset (ox, oy) chosen for its label, return the label anchor's
# position IN DATA SPACE — so a leader line from (x, y) to this point renders
# correctly on the actual data axes. Needed because a crowded point's label
# can end up far enough away (see the 2026-08-06 "Booth" incident: point in a
# dense cluster near a quadrant boundary, label pushed above the dashed
# threshold line with nothing visually tying it back to its point) that it
# reads as belonging to a different point/quadrant entirely without a connector.
function label_anchor_data(x, y, ox, oy; xrange, yrange, fig_w, fig_h, ylog = true)
    sx = (x - xrange[1]) / (xrange[2] - xrange[1]) * fig_w
    yv = ylog ? log10(Float64(y)) : Float64(y)
    sy = (yv - yrange[1]) / (yrange[2] - yrange[1]) * fig_h
    lx, ly = sx + ox, sy + oy
    h    = xrange[1] + lx / fig_w * (xrange[2] - xrange[1])
    ycoord = yrange[1] + ly / fig_h * (yrange[2] - yrange[1])
    (h, ylog ? 10.0 ^ ycoord : ycoord)
end
