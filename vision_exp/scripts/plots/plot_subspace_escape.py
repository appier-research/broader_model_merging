#!/usr/bin/env python3
"""Draw the (e1, e2, escape) trajectory figure from a finished
subspace_escape.py run -- no checkpoints, no GPU, no retraining.

Reads ``results/subspace_escape/full_2task_ta/results.json`` and writes
``trajectory_hero.{png,pdf}`` next to it: one large annotated panel for the
pretrained row on the left, small shape-only thumbnails for the other rows on
the right. Every panel shares one colorbar (average test score) and one
equal-unit scale per axis, so path lengths are comparable across panels.

Each row is subsampled to every 2nd recorded point (plus the last), drawn as
dots joined by straight, flat-colored segments; results.json keeps every point.
The z axis is the row's own escape direction (``z_projection``), so z = 0 is
the task-vector subspace S, tinted on the floor of every box.

    python scripts/plots/plot_subspace_escape.py
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
# Math labels (e1, tau, Delta) in Computer Modern, the LaTeX math font, at
# regular weight exactly as LaTeX sets them. Bundled with matplotlib, no TeX
# needed. (CM bold has no upper-case Greek, so bold math is not an option.)
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# camera-ready submissions reject Type 3 fonts; 42 = TrueType
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

RUN_DIR = "results/subspace_escape/full_2task_ta"

# Panel title per row id. The starting coefficients themselves are left to
# the caption; the title just says what kind of start it is.
ROW_TITLE = {"pretrained": "Pretrained", "random": "Random", "coeff_best": "Best",
             "merged": "Sum", "subspace_gd_coeff": "Subspace GD"}

# Tint of the z=0 plane, i.e. the task-vector subspace S, and of its outline.
S_COLOR = "#49c6fc"


def _display_name(name: str, all_names: Sequence[str]) -> str:
    """Panel title text for a row. ``random_<i>`` is an internal seed index:
    with a single random row it is just "Random"; with several, the index is
    kept ("Random 1") since it is then the only thing telling them apart."""
    m = re.fullmatch(r"random_(\d+)", name)
    if not m:
        return ROW_TITLE.get(name, name)
    n_random = sum(1 for n in all_names if re.fullmatch(r"random_\d+", n))
    return ROW_TITLE["random"] if n_random <= 1 else f"{ROW_TITLE['random']} {m.group(1)}"


def _fill_missing(values: List[Optional[float]]) -> List[float]:
    """Forward/back-fill ``None`` entries so every step gets a real number to
    color by (older results.json files lack the per-point test metrics)."""
    first_real = next((v for v in values if v is not None), 0.0)
    out, last = [], first_real
    for v in values:
        last = v if v is not None else last
        out.append(last)
    return out


def _z(step: dict) -> float:
    # signed component along the row's net escape direction; results.json
    # written before the axis was renamed calls it mtr_projection
    return step.get("z_projection", step.get("mtr_projection", 0.0))


def _stride(traj: List[dict]) -> List[dict]:
    """Keep every 2nd point (plus the last): the drawn path then goes straight
    between the kept points, so what you see is exactly the polyline through
    them."""
    if len(traj) <= 2:
        return traj
    kept = traj[::2]
    if kept[-1] is not traj[-1]:
        kept.append(traj[-1])
    return kept


def _color_scale(trajectories):
    """(cmap, norm, extend) shared across every panel, so a color means the
    same thing in every subplot. The range is the 5th..95th percentile of the
    test scores: one outlier (e.g. the pretrained start at 0.32 while the rest
    sit in 0.88..0.98) would otherwise own a whole end of the scale. Values
    outside saturate, and ``extend`` caps that end of the colorbar with an
    arrow so the clipping is visible."""
    vals = [r.get("test_acc") for traj in trajectories.values()
            for r in traj if r.get("test_acc") is not None]
    extend = "neither"
    if vals:
        lo, hi = min(vals), max(vals)
        if len(vals) > 2:
            lo, hi = (float(np.percentile(vals, 5.0)), float(np.percentile(vals, 100 - 5.0)))
            below, above = any(v < lo for v in vals), any(v > hi for v in vals)
            extend = ("both" if below and above else
                      "min" if below else "max" if above else "neither")
        norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    return plt.get_cmap("viridis"), norm, extend


def _bounds(vals: Sequence[float]) -> Tuple[float, float]:
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 1e-9:
        span = max(abs(hi), 1e-6) * 2
    pad = span * 0.1
    return lo - pad, hi + pad


def _bounds_for(traj: List[dict], spans=None):
    """Per-axis (lo, hi) for one row, at equal data units per axis.

    ``spans``, when given, overrides the computed extents with a fixed size per
    axis while keeping each axes centered on its own data -- that is what makes
    panels comparable: a movement of 0.02 covers the same number of pixels in
    every panel, without forcing panels that sit far apart in S into one box.
    """
    vals = ([r["lambda_hat"][0] for r in traj], [r["lambda_hat"][1] for r in traj],
            [_z(r) for r in traj])
    bounds3 = [_bounds(v) for v in vals]
    # z = 0 is the subspace S itself (the tinted floor), so the box rests on it.
    # Tolerance scaled to the axis: these coordinates come from dot products
    # against fp16 snapshots, so "zero" lands within a few 1e-9 of it.
    z_clamped = False
    tol = 1e-6 * max(abs(min(vals[2])), abs(max(vals[2])), 1e-12)
    if min(vals[2]) >= -tol and bounds3[2][0] < 0:
        bounds3[2] = (0.0, bounds3[2][1])
        z_clamped = True
    if spans is None:
        spans = [max(hi - lo for lo, hi in bounds3)] * 3
    out = []
    for i, (lo, hi) in enumerate(bounds3):
        # The clamped z axis grows upward from 0 rather than re-centering,
        # which would put it back below zero.
        out.append((0.0, spans[i]) if i == 2 and z_clamped
                   else ((lo + hi) / 2 - spans[i] / 2, (lo + hi) / 2 + spans[i] / 2))
    return out


def _draw_start(ax, x, y, z, color, size, label=None):
    """The starting point: a white halo with a black ring under a colored dot,
    so it stands out from the (same-colored) trajectory dots around it."""
    ax.scatter([x], [y], [z], color="white", s=size * 2.6, edgecolors="black",
               linewidths=1.2, zorder=5, depthshade=False)
    ax.scatter([x], [y], [z], color=[color], s=size * 1.2, edgecolors="black",
               linewidths=0.6, zorder=6, depthshade=False)
    if label:
        # Below-left of the marker: the path leaves the start upward and to the
        # right, so this corner is the one nothing else occupies.
        ax.text(x, y, z, f"{label}  ", fontsize=14.0, fontweight="bold", color="#222222",
                zorder=7, ha="right", va="top")


def _draw_subspace(ax, xs, ys, zs, bounds3, line_width, labels=False):
    """Make "leaving S" visible without relying on axis scale: tint the z=0
    plane (that IS the task-vector subspace S), draw the net in-S displacement
    on it (Delta_par), and drop a dashed line from the end point to its shadow
    (Delta_perp) -- the length of that line is how far the model has left S.
    ``labels`` writes those two names next to their lines."""
    (x0, x1), (y0, y1), (z0, _) = bounds3
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    plane = Poly3DCollection([corners], facecolors=S_COLOR, alpha=0.18, edgecolors="none",
                             zorder=0)
    ax.add_collection3d(plane)
    # Opaque outline drawn as its own line ABOVE the pane so it covers the box
    # edge and gridlines underneath instead of reading as a doubled line.
    ring = corners + [corners[0]]
    ax.plot([c[0] for c in ring], [c[1] for c in ring], [z0] * len(ring),
            color=S_COLOR, lw=1.4, solid_capstyle="round", zorder=0.6)
    # net in-S displacement: start -> shadow of the end point
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], [z0, z0], color="#666666",
            lw=max(0.8, line_width * 0.6), zorder=0.5)
    # end point -> its shadow: a thin, quiet dropline with a small anchor where
    # it meets S, so the eye reads "height above the plane" not "another path".
    ax.plot([xs[-1], xs[-1]], [ys[-1], ys[-1]], [z0, zs[-1]], color="#666666",
            lw=0.9, ls=(0, (3, 2.5)), zorder=2)
    ax.scatter([xs[-1]], [ys[-1]], [z0], facecolors="white", edgecolors="#666666",
               s=14, linewidths=0.9, zorder=2.5, depthshade=False)
    if labels:
        # Delta_par: just below the floor segment, 70% of the way from the
        # start so it clears the start marker's halo and its label; a leading
        # half-height blank line pushes the text clear of the line.
        # Delta_perp: to the right of the dropline, halfway up.
        t = 0.7
        ax.text(xs[0] + t * (xs[-1] - xs[0]), ys[0] + t * (ys[-1] - ys[0]), z0,
                "\n" + r"$\Delta_{\parallel}$", fontsize=14.0, color="black",
                ha="center", va="top", linespacing=0.6, zorder=7)
        ax.text(xs[-1], ys[-1], (z0 + zs[-1]) / 2, "  " + r"$\Delta_{\perp}$",
                fontsize=14.0, color="black", ha="left", va="center", zorder=7)


def _draw_row(ax, traj, bounds3, cmap, norm, *, point_size, line_width, marker_size,
              start_label=None, subspace_labels=False):
    """Draw one (already subsampled) row: each segment takes its starting
    point's color, with a dot at every kept point."""
    xs = [r["lambda_hat"][0] for r in traj]
    ys = [r["lambda_hat"][1] for r in traj]
    zs = [_z(r) for r in traj]
    vals = _fill_missing([r.get("test_acc") for r in traj])

    _draw_subspace(ax, xs, ys, zs, bounds3, line_width, labels=subspace_labels)

    if len(xs) >= 2:
        points = list(zip(xs, ys, zs))
        segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
        seg_colors = [cmap(norm(vals[i])) for i in range(len(points) - 1)]
        ax.add_collection3d(Line3DCollection(segments, colors=seg_colors,
                                              linewidths=line_width, zorder=1))

    # depthshade=False on every 3D scatter: mplot3d otherwise fades markers by
    # depth, which makes the same value look like two different colors.
    ax.scatter(xs, ys, zs, color=[cmap(norm(v)) for v in vals], s=point_size,
               edgecolors="none", zorder=3, depthshade=False)
    _draw_start(ax, xs[0], ys[0], zs[0], cmap(norm(vals[0])), marker_size, label=start_label)


def _finish_axes(ax, bounds3, *, zlabel, xlabel=r"$e_1\ (\tau_1)$",
                 ylabel=r"$e_2\ (\tau_2^{\perp})$", zoom=1.0):
    """Limits, a cubic box (equal data units per axis) and labels. ``zoom`` > 1
    enlarges the drawn box inside the same axes area (mplot3d pads heavily)."""
    ax.set_xlim(*bounds3[0])
    ax.set_ylim(*bounds3[1])
    ax.set_zlim(*bounds3[2])
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    # Few ticks: at print size the labels of a dense auto-locator overlap.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.tick_params(labelsize=14.0 * 0.85)
    ax.set_xlabel(xlabel, fontsize=14.0, fontweight="bold", labelpad=14.0 * 0.6)
    ax.set_ylabel(ylabel, fontsize=14.0, fontweight="bold", labelpad=14.0 * 0.6)
    ax.set_zlabel(zlabel, fontsize=14.0, fontweight="bold", labelpad=14.0 * 0.6)


def plot_hero(trajectories: Dict[str, List[dict]], out_path: str) -> None:
    trajectories = {n_: _stride(t) for n_, t in trajectories.items()}
    names = list(trajectories)
    hero = "pretrained"
    small = [n_ for n_ in names if n_ != hero]
    cmap, norm, cbar_extend = _color_scale(trajectories)

    # Per-row bounds first, then adopt the largest span (over rows and axes)
    # so every panel is drawn at one shared, equal-unit scale.
    per_row = {name: _bounds_for(traj) for name, traj in trajectories.items()}
    spans = [max(b[i][1] - b[i][0] for b in per_row.values()) for i in range(3)]
    spans = [max(spans)] * 3
    per_row = {name: _bounds_for(traj, spans=spans) for name, traj in trajectories.items()}
    # Cap the z axis at 3. A panel whose escape exceeds the cap keeps a little
    # headroom above its own data instead, so nothing is drawn outside the box;
    # x/y follow the new z span to keep the box cubic.
    capped = {}
    for name, traj in trajectories.items():
        (x0, x1), (y0, y1), (z0, z1) = per_row[name]
        z1_new = max(3.0, max(_z(r) for r in traj) * 1.03)
        span = z1_new - z0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        capped[name] = [(cx - span / 2, cx + span / 2), (cy - span / 2, cy + span / 2), (z0, z1_new)]
    per_row = capped

    fig = plt.figure(figsize=(6.5, 4.0))
    nrows = max(len(small), 1)
    # Reserve a strip for the colorbar up front (tight_layout leaves it
    # overlapping the 3D panels). Two independent grids so each column gets its
    # own bottom margin: the hero needs room for its x label, the thumbnails
    # do not. The columns overlap a little because mplot3d pads its axes.
    left, right, top = -0.10, 0.93, 0.91
    split = left + (right - left) * 2.0 / (2.0 + 1.0)
    gs_hero = GridSpec(1, 1, figure=fig, left=left, right=split + 0.06, top=top, bottom=0.13)
    gs_thumb = GridSpec(nrows, 1, figure=fig, left=split - 0.06, right=right,
                        top=top, bottom=0.05, hspace=0.4)

    # --- hero: full detail ---------------------------------------------------
    ax = fig.add_subplot(gs_hero[0, 0], projection="3d", computed_zorder=False)
    _draw_row(ax, trajectories[hero], per_row[hero], cmap, norm, point_size=28.0,
              line_width=1.8, marker_size=40.0, start_label="Start", subspace_labels=True)
    _finish_axes(ax, per_row[hero], zlabel=r"$e_3\ (\Delta_{\perp})$")
    ax.set_title(f"Initialization: {_display_name(hero, names).lower()}", fontsize=14.0,
                 fontweight="bold", pad=6)

    # --- thumbnails: shape only ----------------------------------------------
    for i, name in enumerate(small):
        axs = fig.add_subplot(gs_thumb[i, 0], projection="3d", computed_zorder=False)
        _draw_row(axs, trajectories[name], per_row[name], cmap, norm, point_size=28.0 * 0.5,
                  line_width=max(1.0, 1.8 * 0.8), marker_size=40.0 * 0.7)
        _finish_axes(axs, per_row[name], zlabel="", xlabel="", ylabel="", zoom=1.08)
        # Thumbnails show the shape, not coordinates: keep the box and gridlines
        # but drop the tick labels, which are unreadable at this size anyway.
        for axis in (axs.xaxis, axs.yaxis, axs.zaxis):
            axis.set_ticklabels([])
        axs.set_title(_display_name(name, names), fontsize=14.0 * 0.95, fontweight="bold", pad=6)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.845, 0.10, 0.018, 0.78])  # its own strip, not stolen from a panel
    cbar = fig.colorbar(mappable, cax=cax, extend=cbar_extend)
    cbar.set_label("Average score", fontsize=14.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=14.0 * 0.85)
    fig.savefig(out_path, dpi=400)
    fig.savefig(out_path[:-4] + ".pdf")
    plt.close(fig)


def main():
    with open(os.path.join(RUN_DIR, "results.json")) as f:
        data = json.load(f)
    trajectories = {name: row["trajectory"] for name, row in data["rows"].items()}
    out_path = os.path.join(RUN_DIR, "trajectory_hero.png")
    plot_hero(trajectories, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
