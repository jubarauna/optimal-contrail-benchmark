import sys
from pathlib import Path

# False = hide every Python warning (deprecations, cartopy, tight_layout...). Set here,
# before the imports, because matplotlib already warns while it is being imported.
SHOW_WARNINGS = False
if not SHOW_WARNINGS:
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    # A dependency resets the filters on import, so the printer itself is muted too.
    warnings.showwarning = lambda *args, **kwargs: None
    # The "In ...classic.mplstyle: ... deprecated" lines are matplotlib log messages.
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


# UTF-8: the plot helpers print σ / →, which break under Windows' cp1252.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gzip
import pickle

import numpy as np

from src.data import FlightData, GridWeather
from src.paths import FIGURE_DIR
from src.optimization.opentop.drive_opentop import (
    FLIGHTS_FILE, GRID_FILE, load_saved_opt, plot_flight,
)

# ── print size ───────────────────────────────────────────────────────────────
# The Pareto and altitude-profile figures are drawn at the size they are printed, the
# full text width of a two-column paper (7.16 in), with 12 pt type: placed at
# \textwidth, 12 pt here is 12 pt on the page. Placed narrower, the type shrinks with
# the figure -- change PAPER_W rather than scaling the file.
PAPER_W = 7.16          # in
PAPER_PT = 12
PAPER_TITLE_PT = 14     # the figure titles, one step above the rest
import matplotlib.pyplot as _plt
import src.postprocess.plots as _plots
_plots.TEXT_PT = PAPER_PT
_plots.TITLE_PT = PAPER_TITLE_PT
# alt_map starts wider than the page: plot_altitude_2d_map_with_contrails trims the map's
# column to the map's real width, and the figure comes out at about PAPER_W.
_plots.FIGSIZE = {"pareto": (PAPER_W, PAPER_W * 0.62), "altitude": (PAPER_W, PAPER_W * 0.45),
                  "alt_map": (PAPER_W * 1.115, PAPER_W * 0.55)}
_plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": PAPER_PT, "axes.titlesize": PAPER_PT, "axes.labelsize": PAPER_PT,
    "xtick.labelsize": PAPER_PT, "ytick.labelsize": PAPER_PT,
    "legend.fontsize": PAPER_PT, "figure.titlesize": PAPER_PT,
})



# Module-level settings the helpers below read. main() sets them all; these are only
# placeholders so the names exist.
FLIGHT_ID = SIGMA = ROUTE = ALT_MIN = ALT_MAX = SAVE_DIR = None
PARETO_LABELS = True
ALT_MAP_METHOD = "weighted_sum"
ALT_MAP_VALUES = None
ALT_MAP_FUEL_ONLY = True
ALT_MAP_LEGEND_AT = None
ALT_MAP_LEGEND_PT = 10
ALT_MAP_TIME_LEGEND_AT = None


def _abs(path):
    """A relative path is taken from the project root, wherever the script runs from."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _find_pickle(label, folder, suffixes):
    """<flight>_sigma<σ><suffix>.pkl.gz in ``folder`` or one of its direct subfolders
    (m_080/, m_060/), trying ``suffixes`` in order. None if there is none."""
    if not folder.is_dir():
        print(f"  [warning] {label}: {folder} does not exist")
        return None
    dirs = [folder] + sorted(p for p in folder.iterdir() if p.is_dir())
    for suffix in suffixes:
        for d in dirs:
            pkl = d / f"{FLIGHT_ID}_sigma{SIGMA:g}{suffix}.pkl.gz"
            if pkl.exists():
                return pkl
    return None


def _save_dir(save):
    """SAVE_DIR if the figure is to be saved, else None (shown only).

    Created here and not up front: the plot helpers write straight to the path without
    creating the folder, and a run that saves nothing should not leave empty folders.
    """
    if not save:
        return None
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    return SAVE_DIR


def load_graph_payload(pkl):
    """Graph-2D payload from ``pkl``, or None if it was saved with an error.

    A single read point for both the table and the Pareto curve: both need the
    same pickle, and reading it twice would invite the two outputs to diverge
    silently if the path ever changes.
    """
    payload = pickle.load(gzip.open(pkl, "rb"))
    return None if payload.get("error") else payload


MAX_ROWS = 24     # print_table: above this, only the ends are shown


def print_table(label, rows):
    """Results table — the same format for all three solvers.

    ``rows`` is a list of dicts with fuel_kg / contrail / success / method, plus
    the point's label (``coef`` for the weighted sum, ``fuel_slack_frac`` for the
    ε-constraint, ``point`` for the DP_MuO front). OpenTOP returns ``OptResult``
    objects and graph-2D returns the ``summary`` rows, so the caller normalises
    them to dicts first.

    Long fronts are truncated to their ends: DP_MuO returns the whole front —
    124 points on this flight, thousands on others — and dumping it to the terminal
    hides the tables of the other two solvers, which is precisely the comparison at
    hand. The ends suffice because the front is monotone: the extremes are minimum
    fuel and minimum contrail. The figure shows the middle.
    """
    def row(r):
        tag = r.get("coef", r.get("fuel_slack_frac", r.get("point")))
        tag = f"{tag:.3g}" if isinstance(tag, (int, float)) else "-"
        return (f"{str(r.get('method', '')):<22}{tag:>9}{r['fuel_kg']:>12.1f}"
                f"{r['contrail']:>11.2f}{str(r.get('success')):>11}")

    print(f"\n{label}  ({len(rows)} points)")
    print(f"{'method':<22}{'coef/ε':>9}{'fuel [kg]':>12}{'contrail':>11}{'converged':>11}")
    if len(rows) <= MAX_ROWS:
        for r in rows:
            print(row(r))
        return
    half = MAX_ROWS // 2
    for r in rows[:half]:
        print(row(r))
    print(f"{'':<22}{'…':>9}   ({len(rows) - 2 * half} points omitted)")
    for r in rows[-half:]:
        print(row(r))


def fuel_only_series(opt, graph_payloads):
    """Each solver's fuel-only point, as its own series on the Pareto panel.

    It ALWAYS existed in the results, but never appeared in the figure: the
    ``build_df`` of ``plot_pareto_comparison`` requires a label key
    (``coef``, ``fuel_slack_frac``, ``point``) and drops any record that has
    none — and fuel-only has none, because it belongs to no sweep. Here it
    gets the ``sweep`` key and returns to the plot.

    It is worth showing because it is the ANCHOR of the panel: the minimum-fuel end
    of the front, and the F0 from which the ε-constraint derives all of its budgets.

    One series per solver, because the three solve the fuel-only problem on
    different models and do not land in the same place — the distance between
    these points is, by itself, the model comparison.
    """
    # The star in the solver's colour (benchmark palette); no text beside it, the
    # legend already says it is the fuel-only.
    STYLE = {label: dict(color=_benchmark_color(label), marker="*",
                         size=170 if label == "OpenTOP" else 140)
             for label in ("OpenTOP", "DP", "DP_MuO")}
    series = []

    # OpenTOP: identified by meta["method"], not by position in the list. The meta
    # goes to the top of the dict, which is where the helper looks for the labels --
    # EXCEPT the trajectory: with it, build_df takes the branch that reads `grid_cost`
    # from the top of the dict, and there that field only exists as a DataFrame
    # column. Without it, it uses fuel_kg/contrail, which match the trajectory to the
    # third digit.
    ot = [{"fuel_kg": r.fuel_kg, "contrail": r.contrail, "success": r.success,
           **{k: v for k, v in (r.meta or {}).items() if k != "trajectory"}}
          for r in opt.results
          if (r.meta or {}).get("method") == "fuel_only" and r.success]
    if ot:
        series.append({"label": "OpenTOP fuel-only",
                       "records": [{**r, "sweep": "fuel-only"} for r in ot],
                       "label_keys": ("sweep",), "annotate": False,
                       **STYLE["OpenTOP"]})

    # DP / DP_MuO: not every pickle carries the anchor, so it is optional.
    for label, payload in graph_payloads.items():
        items = [r for r in payload["results"].get("fuel_only", []) if r.get("success")]
        if not items:
            continue
        # DP: a star in its benchmark colour with its own legend entry. Others: black,
        # no legend entry ("_nolegend_": matplotlib ignores labels starting with "_").
        if label == "DP":
            name, color = "DP fuel-only", _benchmark_color("DP")
        else:
            name, color = "_nolegend_", "black"
        series.append({"label": name,
                       "records": [{**r, **r.get("meta", {}), "sweep": "fuel-only"}
                                   for r in items],
                       "label_keys": ("sweep",), "annotate": False,
                       **{**STYLE.get(label, dict(marker="*", size=140)), "color": color}})
    return series


def pareto_extra_series(graph_payloads):
    """DP and DP_MuO in the format that ``plot_pareto_comparison`` draws.

    It is not a new figure: these are additional series for the SAME Pareto axes
    that OpenTOP already uses. The records come from ``payload["results"]``, which is
    ``OptResult.as_dict()`` — it already carries ``fuel_kg``/``contrail``, and the
    point's label (``coef``, ``fuel_slack_frac``, ``point``) lives in ``meta``, so it
    has to be lifted to the top of the dict, which is where the helper looks for it.

    DP's ``epsilon_constraint`` sweep is left out: the DP has no constraint, it only
    bisects the weighted-sum ``coef`` until fuel ≈ F0 × slack (model_2D.py:919), so its
    points are weighted-sum solutions under another label, not an ε-constraint.
    """
    # Colour = solver (benchmark palette); marker = method: circle weighted sum,
    # diamond the DP_MuO front.
    STYLE = {
        ("DP", "weighted_sum"):        dict(marker="o", annotate=True),
        ("DP_MuO", "pareto_dp_muo"):   dict(marker="D", size=16),
    }
    METHOD_NAME = {"weighted_sum": " weighted sum", "pareto_dp_muo": ""}
    SKIP = {("DP", "epsilon_constraint")}
    series = []
    for label, payload in graph_payloads.items():
        for sweep, items in payload["results"].items():
            if not items or sweep == "fuel_only" or (label, sweep) in SKIP:
                continue
            style = {"color": _benchmark_color(label),
                     **STYLE.get((label, sweep), dict(marker="^"))}
            series.append({
                "label": f"{PARETO_NAME[label]}{METHOD_NAME.get(sweep, ' ' + sweep)}",
                "records": [{**r, **r.get("meta", {})} for r in items],
                **style,
            })
    return series


def solver_fields(opt, ctx, graph_payloads, gw):
    """The contrail field each solver optimised over, on (distance, altitude).

    Both solvers read the SAME function -- the global bspline for this sigma, shifted to
    the flight's departure (shift_interpolant) -- but they see it differently:

      DP / DP_MuO  ``env.cost_map``: the interpolant sampled once per (route node,
                   flight level), each node at the time the nominal route passes it,
                   and read back by nearest node (get_contrail_cost). Rebuilt with
                   batch_graph2d.build_environment itself, so it is the field the
                   solvers had, not a re-derivation of it.
      OpenTOP      the interpolant itself, continuous in position, altitude and time.
                   Drawn here on a fine mesh (~2 km x 100 ft) along the fuel-only
                   trajectory's track and clock; the other trajectories pass each point
                   a few minutes earlier or later, so for them this is the field up to
                   that time shift.
    """
    import pandas as pd
    sys.path.insert(0, str(PROJECT_ROOT / "2_Run"))
    from batch_graph2d import build_environment
    from src.interpolant import load_global_interpolant

    g = load_global_interpolant(sigma=float(SIGMA), out_dir=Path(GRID_FILE).parent)
    ctx_g = next(iter(graph_payloads.values()))["context"] if graph_payloads else ctx
    meta = {"flight_id": FLIGHT_ID, "actype": ctx["actype"],
            "begin": pd.Timestamp(ctx["begin"]), "end": pd.Timestamp(ctx["end"])}
    built = build_environment(meta, float(SIGMA), Path(FLIGHTS_FILE), gw, float(ctx_g["m0"]),
                              global_interp=g)
    env, interp = built["env"], built["interpolant"]
    # The graph field above is always the bspline (the DP campaign's). OpenTOP's is the
    # shape its own run used -- "linear" under OPENTOP_RUN="linear" -- so it is drawn
    # from that interpolant instead of the bspline it did not see.
    shape = ctx.get("interpolant_shape", "bspline")
    if shape != "bspline":
        from src.interpolant import shift_interpolant
        g_ot, g_ot_meta = load_global_interpolant(sigma=float(SIGMA),
                                                  out_dir=Path(GRID_FILE).parent, shape=shape)
        interp = shift_interpolant(g_ot, ctx["begin"], g_ot_meta)
    fields = {"DP": {"dist": env.distances, "alt": env.altitudes, "cost": env.cost_map,
                     "label": f"DP field: {len(env.distances)} nodes × "
                              f"{env.alt_step_ft:.0f} ft"}}
    fields["DP_MuO"] = {**fields["DP"], "label": fields["DP"]["label"].replace("DP", "DP_MuO", 1)}

    ref = next((r.meta["trajectory"] for r in opt.results
                if (r.meta or {}).get("method") == "fuel_only" and r.success), None)
    if ref is None:
        ref = next(r.meta["trajectory"] for r in opt.results if r.meta.get("trajectory") is not None)
    lat, lon, ts = (ref[c].to_numpy(float) for c in ("latitude", "longitude", "ts"))
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    d = np.concatenate([[0.0], np.cumsum(6371.0 * 2 * np.arcsin(np.sqrt(a)))])
    dist = np.linspace(0.0, d[-1], max(2, int(d[-1] / 2)))
    lo_ft = ALT_MIN if ALT_MIN is not None else ctx["min_alt_ft"]
    hi_ft = ALT_MAX if ALT_MAX is not None else ctx["max_alt_ft"]
    alt = np.arange(lo_ft, hi_ft + 100.0, 100.0)
    D, A = np.meshgrid(dist, alt, indexing="ij")
    X = np.vstack([np.interp(D.ravel(), d, lon), np.interp(D.ravel(), d, lat),
                   A.ravel() * 0.3048, np.interp(D.ravel(), d, ts)])
    cost = np.asarray(interp.map(X.shape[1])(X)).reshape(D.shape)
    fields["OpenTOP"] = {"dist": dist, "alt": alt, "cost": cost,
                         "label": f"OpenTOP field: continuous {shape}"}
    return fields


def plot_graph_altitude(label, payload, *, df_contrails, real, field=None, save=False):
    """DP / DP_MuO altitude profile, using the same helper OpenTOP uses.

    ``plot_altitude_vs_distance_all_trajectories`` already accepts graph-based results
    — it detects ``alt_ft``/``dist_km`` in place of ``trajectory`` (plots.py:1537). Only
    two things are missing that the pickle does not provide ready-made:

    1. the labels (``optimizer``, ``coef``, ``fuel_slack_frac``) live in ``meta``,
       and the helper reads them from the top of the dict, as in OpenTOP's
       ``_results_to_records``;
    2. the route, which the batch does not save (it is rebuildable and heavy). Without
       it the contrail background is not sampled — the helper still draws the curves,
       just without the heat map behind them. Rebuilt here as in batch_graph2d.py:242.
    """
    from src.postprocess.plots import plot_altitude_vs_distance_all_trajectories

    ctx_g = payload["context"]
    route = graph_route(payload, real)

    # One figure per method present in the pickle - NOT a fixed list: DP writes
    # weighted_sum/epsilon_constraint, DP_MuO writes a single pareto_dp_muo,
    # and a new method shows up on its own without touching anything here.
    groups = dict(payload["results"])
    if "fuel_only" in groups and "weighted_sum" in groups:
        # The fuel anchor travels with the weighted sweep, as in plot_flight.
        groups["weighted_sum"] = groups.pop("fuel_only") + groups["weighted_sum"]

    for sweep, items in groups.items():
        if not items:
            continue
        # The labels (optimizer, coef, ...) live in meta; the helper reads the top of the dict.
        recs = [{**r, **r.get("meta", {}), "method": sweep, "objective_type": sweep}
                for r in items]
        plot_altitude_vs_distance_all_trajectories(
            results=recs, df_contrails=df_contrails, route=route, ts_select=0,
            alt_min=ALT_MIN if ALT_MIN is not None else ctx_g["min_alt_ft"],
            alt_max=ALT_MAX if ALT_MAX is not None else ctx_g["max_alt_ft"],
            save_dir=_save_dir(save),
            filename=f"altitude_vs_distance_{label.lower()}_{sweep}.png",
            real_trajectory=real, field=field,
        )
        print(f"  {label}: {sweep} - {len(recs)} trajectories")


def graph_route(payload, real):
    """Lateral route of the graph-2D, which the batch does not save (it is rebuildable and heavy).

    Rebuilt as in batch_graph2d.py:242, over the real trace. Without the real flight it
    returns None: the altitude profile still comes out (just without the contrail
    background), and the map falls back to the great circle in ``graph_records``.
    """
    from openap.aero import ft
    from src.models.grid.contrail_grid import real_flight_route

    if real is None:
        return None
    ctx_g = payload["context"]
    route, _ = real_flight_route(real.flight, mach=ctx_g["mach"],
                                 avg_alt_m=ctx_g["cruise_alt_ft"] * ft,
                                 roc=ctx_g["max_roc_fpm"],
                                 alt_step_ft=ctx_g["alt_step_ft"])
    return route


def _selected(value, chosen):
    """Is ``value`` in ALT_MAP_VALUES? Compared with a tolerance: 0.1 in the config and
    0.1 read from the pickle are not always the same float."""
    if chosen is None:
        return True
    # np.atleast_1d: accepts a bare number — ``(0.5)`` is just 0.5, not a tuple.
    return value is not None and any(np.isclose(value, v) for v in np.atleast_1d(chosen))


def opentop_records(opt):
    """OpenTOP → records for ``plot_altitude_2d_map_with_contrails``.

    The label goes in ``coef`` as text: the helper uses the text as the legend entry,
    and the fuel-only (with neither coef nor ε) would otherwise be left off the map.
    """
    recs = []
    for r in opt.results:
        m = r.meta or {}
        if not r.success or m.get("trajectory") is None:
            continue
        if m.get("method") == "fuel_only":
            if ALT_MAP_FUEL_ONLY:
                # OpenTOP's green, darkened: the same solver as the coefficient line
                # beside it (which gets the plain green or lighter shades), told apart
                # by being the darker one. Solid; the dashed black line is the real flight.
                from matplotlib.colors import to_rgb
                base = np.array(to_rgb(_benchmark_color("OpenTOP")))
                recs.append({"trajectory": m["trajectory"], "coef": "OpenTOP fuel-only",
                             "color": tuple(base * 0.55)})
            continue
        if m.get("method") != ALT_MAP_METHOD:
            continue
        value = m.get("coef") if ALT_MAP_METHOD == "weighted_sum" else m.get("fuel_slack_frac")
        if _selected(value, ALT_MAP_VALUES):
            symbol = "coef" if ALT_MAP_METHOD == "weighted_sum" else "ε"
            recs.append({"trajectory": m["trajectory"],
                         "coef": f"OpenTOP {symbol}={value:g}"})
    return recs


def graph_records(label, payload, real):
    """DP / DP_MuO → records with ``trajectory`` (lat/lon/altitude/ts).

    The pickle only stores ``alt_ft``/``dist_km`` (lat/lon come back None): the 2D path
    is fixed, so lat/lon are interpolated along the route by distance flown.
    """
    import pandas as pd
    from openap.aero import ft
    from src.models.grid.contrail_grid import great_circle_route

    ctx_g = payload["context"]
    route = graph_route(payload, real)
    if route is None:
        print(f"  ({label}: no real flight — map over the great circle)")
        (o_lat, o_lon), (d_lat, d_lon) = ctx_g["origin"], ctx_g["destination"]
        route, _ = great_circle_route(o_lat, o_lon, d_lat, d_lon, mach=ctx_g["mach"],
                                      avg_alt_m=ctx_g["cruise_alt_ft"] * ft,
                                      roc=ctx_g["max_roc_fpm"],
                                      alt_step_ft=ctx_g["alt_step_ft"])
    r_lat, r_lon, r_km = (np.array([p[i] for p in route], dtype=float) for i in range(3))

    def record(item, tag):
        d = np.asarray(item["dist_km"], dtype=float)
        traj = pd.DataFrame({"ts": np.asarray(item["ts_s"], dtype=float),
                             "altitude": np.asarray(item["alt_ft"], dtype=float),
                             "latitude": np.interp(d, r_km, r_lat),
                             "longitude": np.interp(d, r_km, r_lon)})
        return {"trajectory": traj, "coef": tag}

    res = payload["results"]
    recs = []
    if ALT_MAP_FUEL_ONLY:
        recs += [record(it, f"{label} fuel-only")
                 for it in res.get("fuel_only", []) if it.get("success")]

    if label == "DP_MuO":
        front = [it for it in res.get("pareto_dp_muo", []) if it.get("success")]
        if ALT_MAP_VALUES is None:
            idx = np.unique(np.linspace(0, len(front) - 1, min(6, len(front))).round().astype(int))
            front = [front[i] for i in idx]
        else:
            front = [it for it in front if _selected(it["meta"].get("point"), ALT_MAP_VALUES)]
        return recs + [record(it, f"DP_MuO pt {it['meta'].get('point')}") for it in front]

    symbol = "coef" if ALT_MAP_METHOD == "weighted_sum" else "ε"
    for it in res.get(ALT_MAP_METHOD, []):
        if not it.get("success"):
            continue
        m = it.get("meta", {})
        # DP's ε bisects coef (model_2D.py:894): without fuel_slack_frac, label it by coef.
        value = m.get("coef") if ALT_MAP_METHOD == "weighted_sum" else m.get("fuel_slack_frac", m.get("coef"))
        if _selected(value, ALT_MAP_VALUES):
            recs.append(record(it, f"{label} {symbol}={value:g}"))
    return recs


# Each solver's name in the Pareto legend.
PARETO_NAME = {"OpenTOP": "OpenTOP", "DP": "DP", "DP_MuO": "DP MuO"}


def _benchmark_color(label):
    """The solver's colour in the benchmark figures (plots_comparison.METHODS).

    Read from there, not copied: both figures have to keep matching if the palette
    changes. OpenTOP is always the green (opentop_single), as in
    solve_time_distribution — even with OPENTOP_RUN="multistart".
    """
    import matplotlib.pyplot as plt
    # rc_context: importing plots_comparison rewrites plt.rcParams (paper style, large
    # fonts), and that would leak into every figure here.
    with plt.rc_context():
        from plots_comparison import METHODS
    colors = {key: color for key, _, color, _ in METHODS}
    key = {"DP": "dp", "DP_MuO": "dp_muo", "OpenTOP": "opentop_single"}
    return colors[key[label]]


def _flight_title(ctx, real):
    """'DTA653 · B77W · LPPT → FNLU': callsign and aircraft from the data, route from ROUTE."""
    parts = []
    if real is not None and "callsign" in real.flight.data:
        parts.append(str(real.flight.data["callsign"].iloc[0]).strip())
    if ctx.get("actype"):
        parts.append(str(ctx["actype"]).upper())
    if ROUTE is not None:
        parts.append(f"{ROUTE[0]} → {ROUTE[1]}")
    return "  ·  ".join(parts)


def _crop_to_content(fig, pad_in: float = 0.04) -> None:
    """Shrink the canvas to what is drawn on it, keeping every artist's size in inches.

    bbox_inches="tight" only crops the SAVED file; the window plt.show() opens keeps
    the whole canvas, margins included. This crops the canvas itself, so screen and
    file match: the canvas becomes the content's bounding box plus ``pad_in``, and
    every axes (and the suptitle) is moved so that it lands at the same place in
    inches relative to that box.
    """
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())       # inches
    w0, h0 = fig.get_size_inches()
    x0, y0 = bb.x0 - pad_in, bb.y0 - pad_in
    w1, h1 = bb.width + 2 * pad_in, bb.height + 2 * pad_in
    for ax in fig.axes:
        if ax.get_axes_locator() is not None:            # inset axes follow their parent
            continue
        p = ax.get_position()
        ax.set_position([(p.x0 * w0 - x0) / w1, (p.y0 * h0 - y0) / h1,
                         p.width * w0 / w1, p.height * h0 / h1])
    seen = set()      # the suptitle can also be in fig.texts: move each artist once
    for t in [fig._suptitle] + list(fig.texts):
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        tx, ty = t.get_position()
        t.set_position(((tx * w0 - x0) / w1, (ty * h0 - y0) / h1))
    fig.set_size_inches(w1, h1)


def plot_alt_map(by_approach, *, df_contrails, real, alt_min, alt_max, flight_title="",
                 save=False, x_axis="distance"):
    """One altitude + 2D map figure with every approach in ``by_approach``
    ({label: records}), each in its benchmark colour. ``x_axis``: "distance" (km) or
    "time" (minutes since departure) on the altitude panel."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from src.postprocess.plots import plot_altitude_2d_map_with_contrails

    records = []
    for label, recs in by_approach.items():
        if not recs:
            print(f"  {label}: no point matches ALT_MAP_METHOD/ALT_MAP_VALUES")
            continue
        # The solver's colour; extra points come out in lighter shades of it (up to 55%
        # toward white — more than that vanishes over the contrail background).
        # A record that already carries "color" (the fuel-only, dark green) keeps it and uses no shade.
        base = np.array(to_rgb(_benchmark_color(label)))
        shades = iter(np.linspace(0, 0.55, max(1, sum("color" not in r for r in recs))))
        for r in recs:
            records.append(r if "color" in r
                           else {**r, "color": tuple(base + (1 - base) * next(shades))})
        print(f"  {label}: {len(recs)} trajectories")
    if not records:
        return

    # Times New Roman (STIX for maths), PAPER_PT for every text and title. Set
    # for this figure only: the rc_context ends with the function, the sizes and the
    # family stay baked into its artists.
    # The family is named, not "serif": text is laid out when the figure is drawn,
    # after this context has ended, and a generic family would be resolved then.
    with plt.rc_context({"font.family": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "font.size": PAPER_PT, "axes.labelsize": PAPER_PT,
                         "axes.titlesize": PAPER_PT, "xtick.labelsize": PAPER_PT,
                         "ytick.labelsize": PAPER_PT, "legend.fontsize": PAPER_PT,
                         "figure.titlesize": PAPER_PT}):
        plot_altitude_2d_map_with_contrails(results=records, df_contrails=df_contrails,
                                            real_trajectory=real, ts_select=0,
                                            alt_min=alt_min, alt_max=alt_max,
                                            map_legend=False, x_axis=x_axis)
        names = [l for l, r in by_approach.items() if r]
        title = " vs ".join(names)
        fig = plt.gcf()
        # The comparison on the first line, the flight under it.
        sup = fig.suptitle(f"{title}\n{flight_title}" if flight_title else title,
                           fontsize=PAPER_TITLE_PT)
        ax_alt, ax_map = fig.axes[0], fig.axes[1]
        # Legend INSIDE the altitude panel, in an empty zone given in data units
        # (ALT_MAP_LEGEND_AT: km, ft of its lower-left corner) -- where the curves are
        # not depends on the flight, so it is a setting.
        # The time figure has its own corner: ALT_MAP_LEGEND_AT is in km, not minutes.
        legend_at = ALT_MAP_TIME_LEGEND_AT if x_axis == "time" else ALT_MAP_LEGEND_AT
        if ax_alt.get_legend() is not None:
            h, lab = ax_alt.get_legend_handles_labels()
            ax_alt.get_legend().remove()
            anchor = ({"loc": "lower left", "bbox_to_anchor": legend_at,
                       "bbox_transform": ax_alt.transData}
                      if legend_at is not None else {"loc": "best"})
            ax_alt.legend(h, lab, **anchor, frameon=True,
                          fontsize=ALT_MAP_LEGEND_PT,
                          framealpha=0.85, handlelength=1.2, handletextpad=0.4,
                          borderpad=0.25, labelspacing=0.15, borderaxespad=0.0)
        # Room for the title lines plus the map's own title on top, and for tick
        # labels + axis label below (the map's longitudes are rotated). The figure
        # grows by those bands instead of the panels shrinking into them: the map
        # keeps a fixed geographic aspect, so a shorter panel is also a narrower one.
        # subplots_adjust, not tight_layout: tight_layout does not move cartopy axes.
        line_in = PAPER_PT * 1.35 / 72
        top_in = (sup.get_text().count(chr(10)) + 2) * line_in
        bottom_in = 4.0 * line_in
        w_in, h_in = fig.get_size_inches()
        fig.set_size_inches(w_in, h_in + top_in + bottom_in - 0.6)
        h_new = fig.get_size_inches()[1]
        fig.tight_layout()
        fig.subplots_adjust(top=1 - top_in / h_new, bottom=bottom_in / h_new)
        # Same height as the map: the map's frame is set by its aspect, so read where
        # it really landed and give the altitude panel the same vertical extent.
        fig.canvas.draw()
        pm, pa = ax_map.get_position(), ax_alt.get_position()
        ax_alt.set_position([pa.x0, pm.y0, pa.width, pm.height])
        # The panels now sit lower than the band reserved for the title: put the title
        # just above the map's own title instead of at the top of the canvas, and let
        # bbox_inches="tight" drop the empty strip above it.
        sup.set_va("bottom")
        # Above the map's own title as it is actually drawn (its height depends on the
        # backend and dpi, so a fixed offset overlapped it on screen), plus a gap.
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        top = max(ax.title.get_window_extent().transformed(inv).y1
                  for ax in (ax_alt, ax_map) if ax.get_title())
        sup.set_y(top + 0.8 * line_in / h_new)
        _crop_to_content(fig)
        boxes = [a.get_tightbbox().transformed(fig.transFigure.inverted())
                 for a in fig.axes]
        sup.set_x((min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2)
    out_dir = _save_dir(save)
    if out_dir:
        tag = "_time" if x_axis == "time" else ""
        out = out_dir / f"altitude_map_{'_'.join(names).lower()}_{ALT_MAP_METHOD}{tag}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


def main(*, flight_id, sigma, opentop_dir, dp_dir=None, dp_muo_dir=None,
         fuel_only=False, origin=None, destination=None, save=False, show=True,
         with_map=True, pareto_labels=True,
         with_alt_profiles=False, solver_field=True, alt_min=None, alt_max=None,
         contrail_log_floor=1e-6,
         with_alt_map=True, alt_map_approaches=("OpenTOP", "DP"),
         alt_map_method="weighted_sum", alt_map_values=None, alt_map_fuel_only=True,
         alt_map_legend_at=None, alt_map_legend_pt=10,
         with_alt_map_time=True, alt_map_time_legend_at=None):
    """Every table and figure for one flight.

    flight_id            e.g. "424281_06444_0"
    sigma                0 | 1 | 2
    opentop_dir          folder of the OpenTOP pickles (or of the fuel-only ones if
                         fuel_only), e.g. "3_Results/contrail_tol1000_v2/OpenTOP";
                         its m_080/ m_060/ subfolders are searched too
    dp_dir, dp_muo_dir   same for DP and DP_MuO; None = not drawn
    fuel_only            True = opentop_dir holds fuel-only solves (*_fuel_ms / *_fuel)
    origin, destination  airport codes for the titles, or None
    save                 write the figures to 4_Post_Treatment/figures/...
    show                 open the figure windows
    """
    global FLIGHT_ID, SIGMA, ROUTE, ALT_MIN, ALT_MAX, SAVE_DIR, PARETO_LABELS
    global ALT_MAP_METHOD, ALT_MAP_VALUES, ALT_MAP_FUEL_ONLY, ALT_MAP_LEGEND_AT
    global ALT_MAP_LEGEND_PT, ALT_MAP_TIME_LEGEND_AT
    FLIGHT_ID, SIGMA, ALT_MIN, ALT_MAX = flight_id, sigma, alt_min, alt_max
    ROUTE = (origin, destination) if origin and destination else None
    PARETO_LABELS = pareto_labels
    ALT_MAP_METHOD, ALT_MAP_VALUES, ALT_MAP_FUEL_ONLY = alt_map_method, alt_map_values, alt_map_fuel_only
    ALT_MAP_LEGEND_AT, ALT_MAP_LEGEND_PT = alt_map_legend_at, alt_map_legend_pt
    ALT_MAP_TIME_LEGEND_AT = alt_map_time_legend_at
    _plots.CONTRAIL_LOG_FLOOR = contrail_log_floor

    if alt_map_method not in ("weighted_sum", "epsilon_constraint"):
        raise SystemExit(f"alt_map_method={alt_map_method!r} is invalid — "
                         "use 'weighted_sum' or 'epsilon_constraint'")
    invalid = set(alt_map_approaches) - {"OpenTOP", "DP", "DP_MuO"}
    if invalid:
        raise SystemExit(f"alt_map_approaches: {sorted(invalid)} invalid — "
                         "use 'OpenTOP', 'DP' and/or 'DP_MuO'")

    opentop_dir = _abs(opentop_dir)
    pkl = _find_pickle("OpenTOP", opentop_dir,
                       ("_fuel_ms", "_fuel") if fuel_only else ("_ms", ""))
    if pkl is None:
        raise SystemExit(f"no σ={sigma:g} pickle of {flight_id} in {opentop_dir}")
    SAVE_DIR = FIGURE_DIR / (f"{flight_id}_sigma{sigma:g}_{opentop_dir.parent.name}"
                             f"_{opentop_dir.name}")
    print(f"flight={flight_id}  σ={sigma}  reading {pkl}")
    opt, ctx = load_saved_opt(pkl)

    print(f"\n{ctx['actype']}  m0={ctx['m0']}  mach={ctx['mach']:.3f}  "
          f"Altitude {ctx['min_alt_ft']:.0f}–{ctx['max_alt_ft']:.0f} ft")
    print_table("OpenTOP", [{"fuel_kg": r.fuel_kg, "contrail": r.contrail,
                            "success": r.success, **r.meta} for r in opt.results])

    # The graph-2D pickles are read ONCE and reused for the table, the Pareto curve
    # and the altitude profile — all three must describe the same result.
    graph_payloads = {}
    for label, folder, suffix in (("DP", dp_dir, "_dp"), ("DP_MuO", dp_muo_dir, "_dp_muo")):
        if folder is None:
            continue
        gpkl = _find_pickle(label, _abs(folder), (suffix,))
        payload = load_graph_payload(gpkl) if gpkl is not None else None
        if payload is None:
            print(f"\n({label}: no usable result in {folder})")
            continue
        print(f"\n{label}: reading {gpkl}")
        graph_payloads[label] = payload
        print_table(label, payload["summary"])

    # The batch does not save df_cost in the pickle (it is the heavy part), so the
    # contrail background is re-sliced from the same shared grid. Takes ~1 min.
    o_lat, o_lon = ctx["origin"]
    d_lat, d_lon = ctx["destination"]
    gw = GridWeather(GRID_FILE, sigmas=[float(ctx["sigma"])])
    df_cost = gw.slice_for_flight(o_lat, o_lon, d_lat, d_lon, start=ctx["begin"],
                                  stop=ctx["end"], sigma=float(ctx["sigma"]))

    # Actually flown track (dashed black line over the plots) — optional.
    try:
        real = FlightData(start=str(ctx["begin"]), stop=str(ctx["end"]),
                          base_id=ctx["flight_id"], path=str(FLIGHTS_FILE))
    except Exception as exc:
        print(f"  (no real-flight overlay: {exc})")
        real = None

    fields = (solver_fields(opt, ctx, graph_payloads, gw)
              if with_alt_profiles and solver_field else {})

    # All three solvers on the SAME Pareto panel.
    plot_flight(opt, ctx, df_contrails=df_cost, real_trajectory=real,
                with_map=with_map, with_altitude=with_alt_profiles,
                alt_min=alt_min, alt_max=alt_max,
                save_dir=_save_dir(save), show=False,
                contrail_field=fields.get("OpenTOP"),
                pareto_extra=(fuel_only_series(opt, graph_payloads)
                              + pareto_extra_series(graph_payloads)),
                pareto_opentop_style={"color": _benchmark_color("OpenTOP"), "label": "OpenTOP"},
                pareto_annotate=pareto_labels,
                # The ε-constraint front starts at the fuel-only point, as the weighted sum does.
                pareto_eps_anchor=next(((r.fuel_kg, r.contrail) for r in opt.results
                                        if (r.meta or {}).get("method") == "fuel_only"
                                        and r.success), None),
                pareto_title=_flight_title(ctx, real))

    # Same flight, same sigma, same contrail background — only the solver changes.
    if graph_payloads and with_alt_profiles:
        print("\naltitude profile, graph-2D:")
        for label, payload in graph_payloads.items():
            plot_graph_altitude(label, payload, df_contrails=df_cost, real=real,
                                field=fields.get(label), save=save)

    # Altitude + 2D map: every chosen approach on the SAME figure.
    if with_alt_map:
        print(f"\naltitude + 2D map ({alt_map_method}, values={alt_map_values}):")
        a_min = alt_min if alt_min is not None else ctx["min_alt_ft"]
        a_max = alt_max if alt_max is not None else ctx["max_alt_ft"]
        by_approach = {}
        for label in alt_map_approaches:
            if label == "OpenTOP":
                by_approach[label] = opentop_records(opt)
            elif label in graph_payloads:
                by_approach[label] = graph_records(label, graph_payloads[label], real)
            else:
                print(f"  {label}: no result loaded")
        plot_alt_map(by_approach, df_contrails=df_cost, real=real,
                     alt_min=a_min, alt_max=a_max, flight_title=_flight_title(ctx, real),
                     save=save)
        if with_alt_map_time:
            print("altitude + 2D map, time on x:")
            plot_alt_map(by_approach, df_contrails=df_cost, real=real,
                         alt_min=a_min, alt_max=a_max,
                         flight_title=_flight_title(ctx, real), save=save, x_axis="time")

    if show:
        _plt.show()


if __name__ == "__main__":
    main(
        flight_id="4066ab_05279_0",
        sigma=2,                        # 0 | 1 | 2
        opentop_dir="3_Results/contrail_2d_cruise/OpenTOP",
        dp_dir="3_Results/contrail_2d_cruise/DP_retimed",          # None = no DP
        dp_muo_dir="3_Results/contrail_2d_cruise/DP_MuO_retimed",  # None = no DP_MuO
        fuel_only=False,                 # True = opentop_dir holds fuel-only solves
        origin="Origin", destination="Destination",        # for the titles; None = left out
        save=False,
        show=True,
        # ── the rest ──
        with_map=True,                  # cartopy map (slow)
        pareto_labels=True,
        with_alt_profiles=False,
        solver_field=True,
        alt_min=None, alt_max=None,     # ft; None = the optimised window
        contrail_log_floor=None,        # None = linear colour scale
        with_alt_map=True,
        alt_map_approaches=("OpenTOP", "DP"),
        alt_map_method="weighted_sum",  # "weighted_sum" | "epsilon_constraint"
        alt_map_values=(0.5,1.0),          # coef / ε / DP_MuO point; None = all
        alt_map_fuel_only=True,
        alt_map_legend_at=None, # (km, ft); None = best spot
        alt_map_legend_pt=10,
        with_alt_map_time=True,
        alt_map_time_legend_at=None,    # (min, ft); None = best spot
    )
