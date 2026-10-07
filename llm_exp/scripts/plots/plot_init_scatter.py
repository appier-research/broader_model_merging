#!/usr/bin/env python3
"""Scatter: init metric (x) vs weight_gd from that init (y).

Same plot as vision_exp/scripts/plots/plot_init_scatter.py (keep the two plotting
sections in sync); the only substantive differences are the metric names (LLM
tasks score generations, not classification accuracy: ``test_avg_score``
instead of ``test_avg_acc``), the arch titles, and the results layout
(llm_exp writes units directly under <results-dir>/units/, there is no
main_exp/ level).

Merge / pretrained points are read from --results-dir (default results).
Random-init points are read from --random-dir (default
results/weight_gd_random) and averaged over random_42 / random_43 / random_44.

Canonical inits (GD cells come from --gd-strategy, default ``weight_gd_lora``:
the LLM experiments run GD through LoRA, and full-weight GD OOMs on qwen3-4b):
  pretrained -> <gd>.pretrained  (TA unit)
  ta, dare   -> <gd>.avg         (baselines.merged_avg)
  ties, tsvm -> <gd>.merged      (baselines.merged_coeff1)
  random     -> mean of random_<seed> cells

Examples
--------
    python scripts/plots/plot_init_scatter.py --arch qwen3-0.6b
    python scripts/plots/plot_init_scatter.py --arch qwen3-0.6b \\
        --methods pretrained,ta,dare,ties,tsvm,random --metric valid_avg_score
    python scripts/plots/plot_init_scatter.py --arch qwen3-0.6b --gd-strategy weight_gd
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import matplotlib.path
import matplotlib.transforms

DEFAULT_METRIC = "test_avg_score"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_RANDOM_DIR = "results/weight_gd_random"
DEFAULT_TASKS = "bank77,ddxplus,ifeval,usefulness_judge"
DEFAULT_METHODS = "pretrained,ta,dare,ties,tsvm"
DEFAULT_BUDGET = "full"
DEFAULT_GD_STRATEGY = "weight_gd_lora"
GD_STRATEGY_TITLE = {"weight_gd_lora": "", "weight_gd": " (full-weight GD)"}
RANDOM_SEEDS = (42, 43, 44)
UNITS_SUBDIR = "units"
PLOTS_SUBDIR = "plots"

KNOWN_METHODS = ("pretrained", "ta", "dare", "ties", "tsvm", "random")
_TAB10 = plt.get_cmap("tab10").colors  # same method colours as vision_exp
COLOR = {name: _TAB10[i] for i, name in enumerate(KNOWN_METHODS)}

# Included at 0.4*\linewidth in an ICLR paper (\linewidth ~5.5in => ~2.2in).
# The canvas is that printed width, so fonts below equal their on-page point
# size and match the caption without LaTeX rescaling.
PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "font.size": 9, "axes.titlepad": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
}
FIGSIZE = (2.2, 2.0)
MARKER_SIZE = 42
LABEL_FONTSIZE = 6.5
LOW_FRAC = 0.15        # fraction of each axis given to the low (Random) segment
BREAK_FRAC = 0.05      # blank gap between the two segments, where // sits
HIGH_PAD_FRAC = 0.14   # padding around the main cluster (room for labels)
EDGE_KEEPOUT = 0.05    # no tick label this close (axes frac) to the break
COINCIDE_TOL = 0.01    # points closer than this (axes frac) are drawn as wedges
                       # of one disk; ~1/6 of a marker diameter, i.e. offsets the
                       # eye cannot separate. Farther apart -> ordinary overlapping
                       # markers at their own positions.
LOW_ERR_MULT = 1.5     # low segment half-width = this * largest error bar ...
LOW_MIN_HALF = 0.75    # ... but at least this many data units
BREAK_METHODS = {"random"}  # only these may be split off into the low segment
PCT_METRICS = ("acc", "score")  # metrics shown x100 with a (%) axis label
PCT_AXIS_LABELS = ("Init score (%)", "After-GD score (%)")
ARCH_TITLE = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "llama3.2-1b": "Llama-3.2-1B",
}


# name -> (unit method, weight_gd init, baselines cell)
MERGE_SPEC = {
    "pretrained": ("ta", "pretrained", "pretrained"),
    "ta": ("ta", "avg", "merged_avg"),
    "dare": ("dare", "avg", "merged_avg"),
    "ties": ("ties", "merged", "merged_coeff1"),
    "tsvm": ("tsvm", "merged", "merged_coeff1"),
}

LABEL = {
    "pretrained": "Pretrained",
    "ta": "TA",
    "dare": "DARE",
    "ties": "TIES",
    "tsvm": "TSV-M",
    "random": "Random",
}


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _unit_path(results_dir: str, tasks: List[str], arch: str, method: str, budget: str) -> str:
    prefix = f"{len(tasks)}_tasks_{'-'.join(tasks)}"
    return os.path.join(results_dir, UNITS_SUBDIR, f"{prefix}__{arch}__{method}__{budget}.json")


def _load(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _init_metric_key(metric: str) -> str:
    return f"init_{metric}"


def _point_from_merge(
    rec: dict, gd_strategy: str, gd_init: str, baseline_key: str, metric: str,
) -> Optional[Tuple[float, float]]:
    strat = rec.get("strategies") or {}
    gd = (strat.get(gd_strategy) or {}).get(gd_init)
    if not gd or metric not in gd:
        return None
    y = gd[metric]
    init_key = _init_metric_key(metric)
    if init_key in gd:
        return float(gd[init_key]), float(y)
    # Older units predate the init_* fields; the merge-derived inits duplicate
    # a baselines cell, so fall back to that.
    base = (strat.get("baselines") or {}).get(baseline_key)
    if base and metric in base:
        return float(base[metric]), float(y)
    return None


def _point_from_random(rec: dict, gd_strategy: str, metric: str) -> Optional[Tuple[float, float, float, float]]:
    cells = (rec.get("strategies") or {}).get(gd_strategy) or {}
    init_key = _init_metric_key(metric)
    xs, ys = [], []
    for seed in RANDOM_SEEDS:
        cell = cells.get(f"random_{seed}")
        if not cell:
            continue
        x = cell.get(init_key)
        y = cell.get(metric)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0], ys[0], 0.0, 0.0
    return (
        statistics.mean(xs),
        statistics.mean(ys),
        statistics.stdev(xs),
        statistics.stdev(ys),
    )


def collect_points(
    arch: str,
    methods: List[str],
    tasks: List[str],
    budget: str,
    results_dir: str,
    random_dir: str,
    metric: str,
    gd_strategy: str = DEFAULT_GD_STRATEGY,
) -> List[dict]:
    points = []
    for name in methods:
        if name == "random":
            rec = _load(_unit_path(random_dir, tasks, arch, "ta", budget))
            if rec is None:
                print(f"  skip random: no unit in {random_dir}")
                continue
            got = _point_from_random(rec, gd_strategy, metric)
            if got is None:
                print(f"  skip random: no random_* cells with {metric} / init_{metric}")
                continue
            x, y, xerr, yerr = got
            points.append({"name": name, "x": x, "y": y, "xerr": xerr, "yerr": yerr})
            continue

        spec = MERGE_SPEC.get(name)
        if spec is None:
            print(f"  skip {name}: unknown method")
            continue
        unit_method, gd_init, baseline_key = spec
        rec = _load(_unit_path(results_dir, tasks, arch, unit_method, budget))
        if rec is None:
            print(f"  skip {name}: no {unit_method} unit in {results_dir}")
            continue
        got = _point_from_merge(rec, gd_strategy, gd_init, baseline_key, metric)
        if got is None:
            print(f"  skip {name}: missing {gd_strategy}.{gd_init}.{metric} "
                  f"(or baselines.{baseline_key}.{metric})")
            continue
        x, y = got
        points.append({"name": name, "x": x, "y": y, "xerr": 0.0, "yerr": 0.0})
    return points


def _is_pct(metric: str) -> bool:
    return any(s in metric for s in PCT_METRICS)


def _as_display(v: float, metric: str) -> float:
    return 100.0 * v if _is_pct(metric) else v


def _split_gap(vals: List[float], min_gap: float) -> Optional[int]:
    """Index i such that sorted vals split into low = s[:i+1], high = s[i+1:]."""
    if len(vals) < 2:
        return None
    s = sorted(vals)
    gap, idx = max((s[i + 1] - s[i], i) for i in range(len(s) - 1))
    return idx if gap >= min_gap else None


def _segments(
    vals: List[float], errs: List[float], breakable: List[bool],
    min_gap: float, unit: float, nonneg: bool,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
    """Return (high segment, low segment or None) in data units.

    The high segment is the padded range of the main cluster.  The low segment
    (only when a big gap exists and every point below it is `breakable`, i.e.
    Random) is centred on the outlier(s) and just wide enough that their error
    bars stay visible next to the marker.
    """
    idx = _split_gap(vals, min_gap)
    if idx is not None:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        if not all(breakable[i] for i in order[: idx + 1]):
            idx = None
    if idx is None:
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1e-9)
        pad = HIGH_PAD_FRAC * span
        return (max(0.0, lo - pad) if nonneg else lo - pad, hi + pad), None
    s = sorted(range(len(vals)), key=lambda i: vals[i])
    low_i, high_i = s[: idx + 1], s[idx + 1 :]
    lvals = [vals[i] for i in low_i]
    centre = 0.5 * (min(lvals) + max(lvals))
    half = 0.5 * (max(lvals) - min(lvals)) + max(
        LOW_ERR_MULT * max(errs[i] for i in low_i), LOW_MIN_HALF * unit,
    )
    low_lo = max(0.0, centre - half) if nonneg else centre - half
    hvals = [vals[i] for i in high_i]
    lo, hi = min(hvals), max(hvals)
    span = max(hi - lo, 1e-9)
    pad = HIGH_PAD_FRAC * span
    return (lo - pad, hi + pad), (low_lo, centre + half)


class _AxisMap:
    """Map data to [0, 1]; an optional low segment gets LOW_FRAC of the axis."""

    def __init__(self, high: Tuple[float, float], low: Optional[Tuple[float, float]]):
        self.high = high
        self.low = low

    def __call__(self, v: float) -> float:
        hlo, hhi = self.high
        if self.low is None:
            return (v - hlo) / max(hhi - hlo, 1e-9)
        llo, lhi = self.low
        if v <= lhi:
            return LOW_FRAC * (v - llo) / max(lhi - llo, 1e-9)
        t = (v - hlo) / max(hhi - hlo, 1e-9)
        return LOW_FRAC + BREAK_FRAC + t * (1.0 - LOW_FRAC - BREAK_FRAC)

    def break_at(self) -> Optional[float]:
        return None if self.low is None else LOW_FRAC + 0.5 * BREAK_FRAC


def _tick_step(span: float, unit: float) -> float:
    if span <= 5 * unit:
        return 1 * unit
    if span <= 12 * unit:
        return 2 * unit
    if span <= 30 * unit:
        return 5 * unit
    if span <= 60 * unit:
        return 10 * unit
    return 20 * unit


def _ticks(a: float, b: float, step: float) -> List[float]:
    out, t = [], math.ceil(a / step - 1e-9) * step
    while t <= b + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def _axis_ticks(m: _AxisMap, vals: List[float], unit: float) -> List[float]:
    """Ticks in data units; none within EDGE_KEEPOUT of the break."""
    hlo, hhi = m.high
    ticks = _ticks(hlo, hhi, _tick_step(hhi - hlo, unit))
    if m.low is None:
        return ticks
    llo, lhi = m.low
    lo_start = LOW_FRAC + BREAK_FRAC
    ticks = [t for t in ticks if m(t) - lo_start >= EDGE_KEEPOUT]
    # one tick for the outlier: the round number nearest to it, off the break
    outlier = [v for v in vals if v <= lhi]
    cands = [t for t in _ticks(llo, lhi, unit) if LOW_FRAC - m(t) >= 0.6 * EDGE_KEEPOUT]
    if outlier and cands:
        centre = statistics.mean(outlier)
        ticks.insert(0, min(cands, key=lambda t: abs(t - centre)))
    return ticks


def _draw_axis_break(ax, x_at: Optional[float], y_at: Optional[float]) -> None:
    """Mask the spine across the gap and draw a slash at each end of it."""
    kw = dict(transform=ax.transAxes, clip_on=False, solid_capstyle="butt")
    h = 0.5 * BREAK_FRAC
    d = 0.018
    if x_at is not None:
        ax.plot([x_at - h - 0.006, x_at + h + 0.006], [0, 0], color="w", lw=2.4, zorder=5, **kw)
        for c in (x_at - h, x_at + h):
            ax.plot([c - d, c + d], [-1.3 * d, 1.3 * d], color="k", lw=0.8, zorder=6, **kw)
    if y_at is not None:
        ax.plot([0, 0], [y_at - h - 0.006, y_at + h + 0.006], color="w", lw=2.4, zorder=5, **kw)
        for c in (y_at - h, y_at + h):
            ax.plot([-1.3 * d, 1.3 * d], [c - d, c + d], color="k", lw=0.8, zorder=6, **kw)


def _group_coincident(pts: List[Tuple[float, float]], tol: float) -> List[List[int]]:
    """Greedy grouping of points closer than tol (axes fraction)."""
    groups: List[List[int]] = []
    for i, (x, y) in enumerate(pts):
        for g in groups:
            gx, gy = pts[g[0]]
            if math.hypot(x - gx, y - gy) < tol:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


# (dx, dy) in points, ha, va.  Tried in order; first non-colliding wins.
_LABEL_SLOTS = (
    (5, 0, 0.0, 0.5),
    (-5, 0, 1.0, 0.5),
    (4, 4, 0.0, 0.0),
    (4, -4, 0.0, 1.0),
    (0, 5, 0.5, 0.0),
    (0, -5, 0.5, 1.0),
    (-4, 4, 1.0, 0.0),
    (-4, -4, 1.0, 1.0),
)


def _bbox_overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _label_box(parts: List[Tuple[str, str]]):
    """One label made of per-method coloured runs, e.g. TA / DARE."""
    kids = []
    for j, (text, color) in enumerate(parts):
        if j:
            kids.append(TextArea(" / ", textprops=dict(size=LABEL_FONTSIZE, color="0.35")))
        kids.append(TextArea(text, textprops=dict(size=LABEL_FONTSIZE, color=color)))
    return HPacker(children=kids, pad=0, sep=0, align="baseline")


def _place_labels(fig, ax, labels: List[Tuple[float, float, List[Tuple[str, str]]]], marker_boxes) -> None:
    """Direct labels next to points, choosing a slot that avoids markers,
    other labels and the axes edge (measured with the real renderer)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    placed = []
    for px, py, parts in labels:
        best = None
        for dx, dy, ax_al, ay_al in _LABEL_SLOTS:
            t = AnnotationBbox(
                _label_box(parts), (px, py), xycoords="axes fraction",
                xybox=(dx, dy), boxcoords="offset points",
                box_alignment=(ax_al, ay_al), frameon=False, pad=0,
                annotation_clip=False, zorder=4,
            )
            ax.add_artist(t)
            bb = t.get_window_extent(renderer).expanded(1.08, 1.15)
            cost = sum(_bbox_overlap(bb, o) for o in marker_boxes + placed)
            outside = bb.width * bb.height - _bbox_overlap(bb, ax_box)
            cost += 3.0 * outside
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, t, bb)
            else:
                t.remove()
            if cost == 0:
                break
        placed.append(best[2])


def _wedge_marker(k: int, n: int) -> matplotlib.path.Path:
    """k-th of n equal wedges of the unit disk, starting at 12 o'clock.

    Four stray MOVETO vertices at the unit-square corners keep the marker
    scale identical to a plain 'o' regardless of the wedge's own extent.
    """
    a0 = 90.0 + 360.0 * k / n  # counter-clockwise from the top: first wedge is the left half
    w = matplotlib.path.Path.wedge(a0, a0 + 360.0 / n)
    pad_v = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    pad_c = [matplotlib.path.Path.MOVETO] * 4
    return matplotlib.path.Path(
        list(w.vertices) + pad_v, list(w.codes) + pad_c,
    )


def plot_scatter(points: List[dict], out: str, arch: str, metric: str, title: Optional[str]) -> None:
    pct = _is_pct(metric)
    unit = 1.0 if pct else 0.01          # data unit for tick steps
    min_gap = 15.0 * unit                # gap that triggers an axis break
    xs = [_as_display(p["x"], metric) for p in points]
    ys = [_as_display(p["y"], metric) for p in points]
    xe = [_as_display(p["xerr"], metric) for p in points]
    ye = [_as_display(p["yerr"], metric) for p in points]
    brk = [p["name"] in BREAK_METHODS for p in points]
    x_high, x_low = _segments(xs, xe, brk, min_gap, unit, nonneg=pct)
    y_high, y_low = _segments(ys, ye, brk, min_gap, unit, nonneg=pct)
    mx, my = _AxisMap(x_high, x_low), _AxisMap(y_high, y_low)
    pxy = [(mx(x), my(y)) for x, y in zip(xs, ys)]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        labels: List[Tuple[float, float, List[Tuple[str, str]]]] = []
        for g in _group_coincident(pxy, COINCIDE_TOL):
            i0 = g[0]
            px, py = pxy[i0]
            if len(g) == 1:
                c = COLOR[points[i0]["name"]]
                if xe[i0] or ye[i0]:
                    ax.errorbar(
                        px, py,
                        xerr=[[px - mx(xs[i0] - xe[i0])], [mx(xs[i0] + xe[i0]) - px]] if xe[i0] else None,
                        yerr=[[py - my(ys[i0] - ye[i0])], [my(ys[i0] + ye[i0]) - py]] if ye[i0] else None,
                        fmt="o", color=c, ecolor=c, capsize=2, elinewidth=0.8,
                        markersize=MARKER_SIZE ** 0.5, markeredgecolor="white",
                        markeredgewidth=0.4, zorder=3,
                    )
                else:
                    ax.scatter(px, py, s=MARKER_SIZE, color=c, zorder=3,
                               edgecolors="white", linewidths=0.6)
            else:
                # visually coincident points: each drawn as one colour wedge of
                # a disk, at its own true position (they differ by less than
                # COINCIDE_TOL).  Each wedge keeps its white edge so the seam
                # shows they are distinct points.  Slightly larger to make up
                # for the seam.
                for k, i in enumerate(g):
                    qx, qy = pxy[i]
                    if xe[i] or ye[i]:
                        ax.errorbar(
                            qx, qy,
                            xerr=[[qx - mx(xs[i] - xe[i])], [mx(xs[i] + xe[i]) - qx]] if xe[i] else None,
                            yerr=[[qy - my(ys[i] - ye[i])], [my(ys[i] + ye[i]) - qy]] if ye[i] else None,
                            fmt="none", ecolor=COLOR[points[i]["name"]], capsize=2,
                            elinewidth=0.8, zorder=2.9,
                        )
                    ax.scatter(
                        qx, qy, s=1.15 * MARKER_SIZE, marker=_wedge_marker(k, len(g)),
                        color=COLOR[points[i]["name"]], zorder=3,
                        edgecolors="white", linewidths=0.6,
                    )
            for i in g:  # one label per point, anchored at its own position
                labels.append((*pxy[i], [(LABEL[points[i]["name"]], COLOR[points[i]["name"]])]))

        xticks = _axis_ticks(mx, xs, unit)
        yticks = _axis_ticks(my, ys, unit)
        ax.set_xticks([mx(t) for t in xticks], [f"{t:g}" for t in xticks])
        ax.set_yticks([my(t) for t in yticks], [f"{t:g}" for t in yticks])
        if pct:
            ax.set_xlabel(PCT_AXIS_LABELS[0])
            ax.set_ylabel(PCT_AXIS_LABELS[1])
        else:
            ax.set_xlabel(f"Init {metric}")
            ax.set_ylabel(f"After GD {metric}")
        ax.set_title(title or ARCH_TITLE.get(arch, arch))
        fig.tight_layout(pad=0.3)
        _draw_axis_break(ax, mx.break_at(), my.break_at())

        # marker boxes in display coords for label collision checks
        r = 0.5 * MARKER_SIZE ** 0.5 * fig.dpi / 72.0 * 1.6
        boxes = []
        for px, py in pxy:
            X, Y = ax.transAxes.transform((px, py))
            boxes.append(matplotlib.transforms.Bbox([[X - r, Y - r], [X + r, Y + r]]))
        _place_labels(fig, ax, labels, boxes)

        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=400)
        fig.savefig(os.path.splitext(out)[0] + ".pdf")
        plt.close(fig)
    print(f"Wrote {out}")


def _default_out(arch: str, tasks: List[str], budget: str, gd_strategy: str) -> str:
    prefix = f"{len(tasks)}_tasks_{'-'.join(tasks)}"
    tag = "" if gd_strategy == DEFAULT_GD_STRATEGY else f"__{gd_strategy}"
    return os.path.join("results", "init_scatters", PLOTS_SUBDIR, f"{prefix}__{arch}__{budget}{tag}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, help="Arch tag, e.g. qwen3-0.6b")
    p.add_argument(
        "--methods", default=DEFAULT_METHODS,
        help=f"Comma list from {list(KNOWN_METHODS)} (default: {DEFAULT_METHODS})",
    )
    p.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated task labels")
    p.add_argument("--budget", default=DEFAULT_BUDGET, help="Budget label in the unit filename")
    p.add_argument(
        "--gd-strategy", default=DEFAULT_GD_STRATEGY,
        help="Strategy block holding the GD cells (weight_gd_lora or weight_gd)",
    )
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Results root holding units/")
    p.add_argument("--random-dir", default=DEFAULT_RANDOM_DIR, help="Side root for random_* weight_gd")
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    methods = _parse_list(args.methods)
    unknown = [m for m in methods if m not in KNOWN_METHODS]
    if unknown:
        p.error(f"unknown --methods {unknown}; expected subset of {list(KNOWN_METHODS)}")
    tasks = _parse_list(args.tasks)
    if not tasks:
        p.error("--tasks is empty")

    print(f"arch={args.arch}  methods={methods}  metric={args.metric}")
    points = collect_points(
        args.arch, methods, tasks, args.budget, args.results_dir, args.random_dir, args.metric,
        gd_strategy=args.gd_strategy,
    )
    if not points:
        raise SystemExit("No points to plot (missing units or cells)")
    out = args.out or _default_out(args.arch, tasks, args.budget, args.gd_strategy)
    title = args.title
    if title is None:
        suffix = GD_STRATEGY_TITLE.get(args.gd_strategy, f" ({args.gd_strategy})")
        title = ARCH_TITLE.get(args.arch, args.arch) + suffix
    plot_scatter(points, out, args.arch, args.metric, title)


if __name__ == "__main__":
    main()
