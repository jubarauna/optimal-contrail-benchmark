import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openap import aero, FuelFlow
from openap.aero import ft, kts
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from cartopy.feature import OCEAN, LAND, BORDERS
from scipy.interpolate import RegularGridInterpolator

from src.models.grid.contrail_grid import ContrailEnvironmentFromGrid
import tempfile
from src.data import FlightData
from src.paths import DATA_DIR, RUN_DIR
from src.models.grid.contrail_grid import ContrailEnvironmentFromGrid
from openap.aero import ft
# Tools -----------------------------------------------------------------------------------------------------------------

# Paper mode for the Pareto and altitude-profile figures. None keeps the screen sizes
# below; a caller preparing figures for print sets TEXT_PT (every text of those two
# figures, annotations and legends included) and FIGSIZE (the printed size, so that
# TEXT_PT points in the file are TEXT_PT points on the page).
TEXT_PT = None
TITLE_PT = None         # the Pareto figure's title; None = TEXT_PT (or its own 12)
FIGSIZE = {"pareto": (9, 6), "altitude": (14, 6), "alt_map": (14, 5)}
# Colour scale of the contrail background in the altitude profiles (and the map next
# to them). None = linear from 0; a value = log scale from it, cost below it left white,
# so a residual of 1e-3 is as visible as the core. Set by the caller, like TEXT_PT.
CONTRAIL_LOG_FLOOR = None


def _pt(default):
    """TEXT_PT when the caller set one, else this call site's own size."""
    return TEXT_PT if TEXT_PT is not None else default


def _contrail_norm(vmin, vmax):
    """Colour norm of a contrail background: LogNorm from CONTRAIL_LOG_FLOOR when set,
    else linear over [vmin, vmax]."""
    from matplotlib.colors import LogNorm, Normalize
    if CONTRAIL_LOG_FLOOR is not None:
        return LogNorm(vmin=CONTRAIL_LOG_FLOOR, vmax=max(vmax, CONTRAIL_LOG_FLOOR * 10))
    return Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)


def _contrail_values(values):
    """Values below the log floor masked (drawn white); unchanged on a linear scale."""
    if CONTRAIL_LOG_FLOOR is None:
        return values
    return np.ma.masked_less(np.asarray(values, dtype=float), CONTRAIL_LOG_FLOOR)


def _contrail_cbar_format(cbar):
    """Two decimals on a linear scale; a log scale keeps its 10^k ticks."""
    if CONTRAIL_LOG_FLOOR is None:
        from matplotlib.ticker import FuncFormatter
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _enclosing_alt_levels(dfc, alt_min, alt_max):
    """Widen [alt_min, alt_max] to the contrail-grid levels that ENCLOSE it.

    The optimization window rarely lands on a grid level (h_min = cruise − 4000 ft
    gives e.g. 35025 ft against 1000 ft levels), so keeping only the levels
    strictly inside it drops the level just below (35000 ft) — and the band the
    trajectories actually fly in is left unpainted. Returns the (lo, hi) level
    bounds to slice on; the axes still clip the view to alt_min/alt_max.
    """
    levels = np.sort(dfc["altitude"].unique())
    below = levels[levels <= alt_min]
    above = levels[levels >= alt_max]
    lo = below.max() if len(below) else levels.min()
    hi = above.min() if len(above) else levels.max()
    return lo, hi


def _real_trajectory_df(obj):
    """Normalise a real flown trajectory to a DataFrame for plotting.

    Accepts a ``FlightData`` instance, a traffic ``Flight``/``Traffic``, or a
    plain DataFrame. Returns a copy with a numeric ``ts`` column (seconds from
    the first sample) and, where possible, a ``tas`` column (real data usually
    only carries ``groundspeed``, which is used as a fallback).
    """
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif getattr(obj, "flight", None) is not None:          # FlightData
        f = obj.flight
        df = (f.data if hasattr(f, "data") else f).copy()
    elif hasattr(obj, "data"):                              # Flight / Traffic
        df = obj.data.copy()
    else:
        raise TypeError(
            "real_trajectory must be a FlightData, traffic Flight/Traffic, or "
            f"DataFrame with latitude/longitude/altitude; got {type(obj)!r}"
        )

    if "ts" not in df.columns:
        t = pd.to_datetime(df["timestamp"], utc=True)
        df["ts"] = (t - t.iloc[0]).dt.total_seconds()
    if "tas" not in df.columns and "groundspeed" in df.columns:
        df["tas"] = df["groundspeed"]
    return df.sort_values("ts").reset_index(drop=True)


# A* plots -----------------------------------------------------------------------------------------------------------------
def plot_cost_map(results, paths=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    if isinstance(results, ContrailEnvironmentFromGrid):
        env_distances = results.distances
        env_altitudes = results.altitudes
        env_cost_map  = results.cost_map
    else:
        env_distances = results['env_distances']
        env_altitudes = results['env_altitudes']
        env_cost_map  = results['env_cost_map']
    X, Y = np.meshgrid(env_distances, env_altitudes)
    cm = ax.pcolormesh(X, Y, env_cost_map.T, shading="auto", cmap="viridis")
    plt.colorbar(cm, ax=ax, label="Contrail Cost")

    # Default colors and styles for known labels, auto-assign for extras
    default_colors = {"Baseline": "gray", "A*": "cyan", "OpenAP": "orange"}
    auto_colors = plt.cm.tab10.colors  # 10 distinct colors for unknown labels
    auto_idx = 0

    if paths:
        for label, trajectory in paths.items():
            # Accept either (dists, alts) tuple or list of (dist, alt, ts) nodes
            if isinstance(trajectory, tuple) and len(trajectory) == 2:
                dists, alts = trajectory
            else:
                # List of (dist, alt, ts) path nodes from A*
                dists = [p[0] for p in trajectory]
                alts  = [p[1] for p in trajectory]

            if label in default_colors:
                color = default_colors[label]
            else:
                color = auto_colors[auto_idx % len(auto_colors)]
                auto_idx += 1

            linestyle = "--" if label == "Baseline" else "-"
            ax.plot(dists, alts, linewidth=2.5, label=label,
                    color=color, linestyle=linestyle)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title("Contrail Cost Map with Trajectories")
    ax.legend()
    plt.tight_layout()

# OpenTOP plots -----------------------------------------------------------------------------------------------------------------
def _declutter_annotations(ax, texts, step_pt=8, max_tries=12):
    """Nudge annotations vertically until none overlaps an earlier one.

    Coincident sweep points (e.g. coef 0.9 and 1.0 landing on the same trajectory)
    otherwise print their labels on top of each other. Each label keeps its x and is
    moved up/down in ``step_pt`` increments (alternating), first fit wins. Labels
    are also raised above the markers.
    """
    if not texts:
        return
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    placed = []
    for t in texts:
        t.set_zorder(6)   # above every marker, so a dense front cannot hide a label
        # The offset the caller asked for, kept so a later pass (restyling at another
        # text size) can start again from it instead of from this pass's nudges.
        if not hasattr(t, "_orig_xyann"):
            t._orig_xyann = t.xyann if hasattr(t, "xyann") else (0, 0)
        x0, y0 = t._orig_xyann
        for k in range(max_tries + 1):
            dy = 0 if k == 0 else step_pt * ((k + 1) // 2) * (1 if k % 2 else -1)
            t.xyann = (x0, y0 + dy)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.1)
            if not any(bb.overlaps(o) for o in placed):
                break
        placed.append(t.get_window_extent(renderer))


def plot_pareto_comparison(results_weighted, results_epsilon, save_dir=None,
                           extra_series=None, opentop_style=None, title=None, show=True,
                           annotate=True, eps_anchor=None):
    """
    Plot Pareto fronts for both weighted sum and epsilon constraint on the same graph.

    results_weighted: list of dicts with "coef" key
    results_epsilon: list of dicts with "fuel_slack_frac" (fuel factor) key
    extra_series: optional list of dicts, each drawn on the SAME axes with the same
        marks — ``{"label", "records", "color", "marker", "label_keys", "annotate"}``.
        This is how the graph-2D solvers (DP, DP_MuO) join the OpenTOP figure
        instead of getting one of their own: the comparison only means anything if
        every solver is read off one pair of axes.
    opentop_style: optional ``{"color", "label"}`` for the OpenTOP sweeps. Both
        sweeps then share that colour (the marker tells them apart: circle =
        weighted sum, square = ε-constraint) and the legend reads
        "<label> weighted sum" / "<label> ε-constraint". Default: the original
        blue/red "Weighted sum" / "ε-constraint".
    eps_anchor: optional (fuel, contrail) of the fuel-only point. The ε-constraint
        sweep never contains it (its budgets are fractions ABOVE that fuel), so its
        front line otherwise starts at the first budget; given, the line starts at
        the fuel-only point, like the weighted-sum one.
    annotate: write each point's sweep value beside it (the coefficient, "ε=1.01").
        False draws the points bare, extra series included, whatever their own
        ``"annotate"`` says.
    title: optional title override (may be multi-line, e.g. a flight line on top).

    The dashed Pareto lines are not in the legend (the markers already name each
    series), and overlapping point labels are nudged apart.
    """

    def build_df(results, *label_keys):
        rows = []
        for r in results:
            key = next((k for k in label_keys if r.get(k) is not None), None)
            if key is None or str(r[key]) in ("great circle"):
                continue
            # Two record shapes reach this function. OpenTOP carries the full
            # trajectory, so fuel is the mass burnt over it and the grid cost is the
            # integral along it; the graph-2D solvers store the two totals directly
            # (batch_graph2d writes OptResult.as_dict()). Prefer the trajectory when
            # it is there, fall back to the totals otherwise.
            if r.get("trajectory") is not None:
                fuel = float(np.sum(r["trajectory"].mass.iloc[0]
                                    - r["trajectory"].mass.iloc[-1]))
                grid = float(np.sum(r["grid_cost"]))
            else:
                fuel = float(r["fuel_kg"])
                grid = float(r["contrail"])
            rows.append({
                "success": r.get("solver_success", r.get("success", True)),
                "obj": r["total_objective"][-1] if r.get("total_objective") is not None else np.nan,
                "iterations": r.get("solver_iterations"),
                "label": r[key],
                "label_key": key,
                "fuel_cost": fuel,
                "grid_cost": grid,
                "sigma": r.get("smooth_factor"),
            })
        # Named columns even when empty: a run without an epsilon sweep would
        # otherwise give a column-less frame and a KeyError on df["success"].
        return pd.DataFrame(rows, columns=["success", "obj", "iterations", "label",
                                           "label_key", "fuel_cost", "grid_cost", "sigma"])

    def get_pareto(valid):
        costs = valid[["fuel_cost", "grid_cost"]].to_numpy(dtype=float)
        pareto_mask = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            others = np.delete(costs, i, axis=0)
            if np.any(np.all(others <= c, axis=1) & np.any(others < c, axis=1)):
                pareto_mask[i] = False
        return valid[pareto_mask].sort_values("fuel_cost")

    df_w = build_df(results_weighted, "coef")
    # Label ε points with the fuel factor (ε = budget / F0); "fuel_max" is the
    # last-resort fallback for older records that only stored the budget in kg.
    df_e = build_df(results_epsilon, "fuel_slack_frac", "epsilon_factor", "fuel_max")

    # astype(bool): an EMPTY frame's "success" is object-typed, and df[object Series]
    # selects columns instead of rows -- a run without an epsilon sweep then loses
    # every column and get_pareto raises a KeyError.
    ok_w, ok_e = df_w["success"].astype(bool), df_e["success"].astype(bool)
    valid_w, valid_e = df_w[ok_w], df_e[ok_e]
    failed_w, failed_e = df_w[~ok_w], df_e[~ok_e]

    pareto_w = get_pareto(valid_w)
    pareto_e = get_pareto(valid_e if eps_anchor is None else pd.concat(
        [valid_e[["fuel_cost", "grid_cost"]],
         pd.DataFrame({"fuel_cost": [eps_anchor[0]], "grid_cost": [eps_anchor[1]]})],
        ignore_index=True))

    sigma = (df_w["sigma"].iloc[0] if len(df_w) > 0
             else df_e["sigma"].iloc[0] if len(df_e) > 0 else None)

    fig, ax = plt.subplots(figsize=FIGSIZE["pareto"])

    if opentop_style:
        c_w = c_e = opentop_style["color"]
        lab_w = f"{opentop_style['label']} weighted sum"
        lab_e = f"{opentop_style['label']} ε-constraint"
    else:
        c_w, c_e, lab_w, lab_e = "steelblue", "tomato", "Weighted sum", "ε-constraint"
    notes = []   # every point label, decluttered together at the end

    # --- Scatter points ---
    ax.scatter(failed_w["fuel_cost"], failed_w["grid_cost"],
               color="gray", marker="x", alpha=0.5, zorder=2)
    ax.scatter(failed_e["fuel_cost"], failed_e["grid_cost"],
               color="gray", marker="x", alpha=0.5, zorder=2)

    if len(valid_w):
        ax.scatter(valid_w["fuel_cost"], valid_w["grid_cost"],
                   color=c_w, marker="o", s=60, label=lab_w, alpha=0.85, zorder=3)
    if len(valid_e):
        ax.scatter(valid_e["fuel_cost"], valid_e["grid_cost"],
                   color=c_e, marker="s", s=60, label=lab_e, alpha=0.85, zorder=3)

    # --- Pareto fronts (not in the legend: the markers already name the series) ---
    ax.plot(pareto_w["fuel_cost"], pareto_w["grid_cost"],
            color=c_w, linestyle="--", linewidth=2, zorder=4)
    ax.plot(pareto_e["fuel_cost"], pareto_e["grid_cost"],
            color=c_e, linestyle="--", linewidth=2, zorder=4)

    # --- Annotations ---
    for _, row in (valid_w.iterrows() if annotate else ()):
        notes.append(ax.annotate(str(row["label"]), (row["fuel_cost"], row["grid_cost"]),
                                 fontsize=_pt(7), color=c_w,
                                 xytext=(0, 6), textcoords="offset points", ha="center"))

    for _, row in (valid_e.iterrows() if annotate else ()):
        if row["label_key"] == "fuel_max":       # no factor stored — show the budget
            text = f"{float(row['label']):.0f} kg"
        else:
            text = f"ε={float(row['label']):.2f}"
        notes.append(ax.annotate(text, (row["fuel_cost"], row["grid_cost"]),
                                 fontsize=_pt(7), color=c_e,
                                 xytext=(0, -10), textcoords="offset points", ha="center"))

    # --- Other solvers, same axes, same marks ---
    # Drawn after the OpenTOP series and annotated only on request: a full DP_MuO
    # front is hundreds of points, and labelling them would bury the figure.
    for serie in (extra_series or []):
        df_x = build_df(serie["records"], *serie.get("label_keys", ("coef", "fuel_slack_frac", "point")))
        if df_x.empty:
            continue
        ok = df_x[df_x["success"]]
        color, marker = serie.get("color", "seagreen"), serie.get("marker", "^")
        ax.scatter(ok["fuel_cost"], ok["grid_cost"], color=color, marker=marker,
                   s=serie.get("size", 45), label=serie["label"], alpha=0.85, zorder=3)
        # A single-point series (the fuel-only anchor) has no front to draw.
        front = get_pareto(ok)
        if len(front) > 1:
            ax.plot(front["fuel_cost"], front["grid_cost"], color=color,
                    linestyle="--", linewidth=1.6, zorder=4)
        if annotate and serie.get("annotate"):
            for _, row in ok.iterrows():
                # The label is a sweep parameter for the sweeps (numeric) but a plain
                # name for the fuel-only anchor, which has neither coef nor epsilon.
                lab = row["label"]
                text = f"{float(lab):.3g}" if isinstance(lab, (int, float, np.floating)) else str(lab)
                notes.append(ax.annotate(text,
                                         (row["fuel_cost"], row["grid_cost"]), fontsize=_pt(7),
                                         color=color, xytext=(0, 6), textcoords="offset points",
                                         ha="center"))

    ax.set_xlabel("Fuel cost (kg)", fontsize=_pt(11))
    ax.set_ylabel("Contrail cost", fontsize=_pt(11))
    ax.set_title(title if title is not None else
                 fr"Pareto Front Comparison: Weighted Sum vs $\varepsilon$-Constraint, $\sigma={sigma}$",
                 fontsize=TITLE_PT or _pt(12))
    ax.legend(fontsize=_pt(9))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _declutter_annotations(ax, notes)
    # Saved BEFORE plt.show(): with a blocking backend, closing the window destroys the
    # figure, and a savefig after it writes a blank one.
    if save_dir:
        out_path = os.path.join(save_dir, "pareto_comparison.png")
        fig.savefig(out_path, dpi=300)
        print(f"📈 Saved to {out_path}")
    # show=False leaves the window to the caller's own plt.show(), so the Pareto
    # opens together with the other figures instead of blocking before them.
    if show:
        plt.show()

    return fig, ax

def _haversine_cum(lat, lon):
    """Cumulative great-circle distance (km) along a lat/lon polyline."""
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return np.concatenate([[0.0], np.cumsum(6371.0 * 2 * np.arcsin(np.sqrt(a)))])


def _prepare_trajectories_for_contrail_plot(results, windfield, filter_fuel_max, real_trajectory):
    """Shared setup of the *_with_contrails plots: filter results, pick colours,
    distances along each track, the optional real flight and the reference track."""
    results = results or []
    if filter_fuel_max is not None:
        results = [r for r in results if r.get("fuel_max") in filter_fuel_max]

    # Extract the trajectories
    windfield = results[1]["wind"] if (windfield and len(results) > 1) else None
    trajectories = [r["trajectory"] for r in results]
    # A record may carry its own "color" (e.g. one colour family per solver).
    default_colors = cm.tab20(range(len(trajectories) + 1))
    colors = [r.get("color", default_colors[i]) for i, r in enumerate(results)]
    traj_dists = [_haversine_cum(df["latitude"].values, df["longitude"].values)
                  for df in trajectories]

    # Optional real flown trajectory (overlaid in black dashed on every panel)
    real_df = _real_trajectory_df(real_trajectory) if real_trajectory is not None else None
    real_dist = (_haversine_cum(real_df["latitude"].values, real_df["longitude"].values)
                 if real_df is not None else None)

    # Reference trajectory (optimized if any, else the real one) drives the map
    # projection and the contrail-background x-axis. At least one is required.
    ref_df = trajectories[0] if trajectories else real_df
    if ref_df is None:
        raise ValueError("Nothing to plot: pass `results` and/or `real_trajectory`.")

    return results, windfield, trajectories, colors, traj_dists, real_df, real_dist, ref_df


def _draw_altitude_contrail_background(ax_alt, df_contrails, ref_df, ts_select, alt_min, alt_max, cmap,
                                       colorbar=True, alpha=0.6, x_ref=None, follow_time=False):
    """Contrail cost along the reference track vs altitude, as a pcolormesh on ax_alt
    (same logic as plot_altitude_vs_distance_all_trajectories).

    Returns the pcolormesh (None if there is nothing to draw); ``colorbar=False``
    leaves the colorbar to the caller, e.g. one shared with the map.
    ``x_ref``: x coordinate of each point of ``ref_df`` (e.g. its time); None = the
    distance along it.
    ``follow_time``: sample each point of ``ref_df`` at ITS time (``ref_df["ts"]``,
    linear in time between grid hours) instead of the one grid hour nearest
    ``ts_select``. The background then shows the field the reference trajectory
    flew through; ``ts_select`` is not read.
    """
    dfc = df_contrails.copy()
    lo, hi = _enclosing_alt_levels(dfc, alt_min, alt_max)
    in_alt = (dfc["altitude"] >= lo) & (dfc["altitude"] <= hi)
    grid_ts = np.sort(dfc["ts"].unique())

    if follow_time:
        ref_ts = ref_df["ts"].to_numpy(float)
        # The grid hours spanning the flight, plus the one on each side that brackets it.
        i0 = max(int(np.searchsorted(grid_ts, ref_ts.min(), side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(grid_ts, ref_ts.max(), side="left")), len(grid_ts) - 1)
        times = grid_ts[i0:i1 + 1]
        dfc_slice = (
            dfc[dfc["ts"].isin(times) & in_alt]
            .groupby(["ts", "latitude", "longitude", "altitude"], as_index=False)
            .agg({"cost": "max"})
        )
    else:
        chosen_ts = (
            grid_ts[np.argmin(np.abs(grid_ts - ts_select))]
            if ts_select is not None else grid_ts[len(grid_ts) // 2]
        )
        dfc_slice = (
            dfc[(dfc["ts"] >= chosen_ts - 1800) & (dfc["ts"] <= chosen_ts + 1800) & in_alt]
            .groupby(["latitude", "longitude", "altitude"], as_index=False)
            .agg({"cost": "max"})
        )

    ref_lats = ref_df["latitude"].values
    ref_lons = ref_df["longitude"].values
    ref_dists = _haversine_cum(ref_lats, ref_lons)

    alt_levels = np.array(sorted(dfc_slice["altitude"].unique()), dtype=float)
    if len(alt_levels) < 2:
        return None

    cost_matrix = np.zeros((len(ref_dists), len(alt_levels)))

    for j, alt in enumerate(alt_levels):
        layer = dfc_slice[dfc_slice["altitude"] == alt]
        if layer.empty:
            continue

        lats_g = np.sort(layer["latitude"].unique())
        lons_g = np.sort(layer["longitude"].unique())
        if len(lats_g) < 2 or len(lons_g) < 2:
            continue

        if follow_time:
            cube = (
                layer.pivot_table(index=["ts", "latitude"], columns="longitude",
                                  values="cost", aggfunc="mean")
                .reindex(index=pd.MultiIndex.from_product([times, lats_g]), columns=lons_g)
                .fillna(0)
                .to_numpy()
                .reshape(len(times), len(lats_g), len(lons_g))
            )
            t_axis = times
            if len(times) < 2:   # the interpolator needs two points per axis
                t_axis, cube = np.r_[times, times + 1.0], np.concatenate([cube, cube])
            interp_layer = RegularGridInterpolator(
                (t_axis, lats_g, lons_g), cube,
                method="linear", bounds_error=False, fill_value=0.0
            )
            # Clipped in time: before the first / after the last hour, hold it.
            pts = np.column_stack([np.clip(ref_ts, t_axis[0], t_axis[-1]), ref_lats, ref_lons])
        else:
            pivot = (
                layer.pivot_table(index="latitude", columns="longitude",
                                  values="cost", aggfunc="mean")
                .reindex(index=lats_g, columns=lons_g)
                .fillna(0)
            )
            interp_layer = RegularGridInterpolator(
                (lats_g, lons_g), pivot.values,
                method="linear", bounds_error=False, fill_value=0.0
            )
            pts = np.column_stack([ref_lats, ref_lons])

        cost_matrix[:, j] = np.maximum(0, interp_layer(pts))

    X, Y = np.meshgrid(alt_levels, ref_dists if x_ref is None else np.asarray(x_ref, dtype=float))

    pcm = ax_alt.pcolormesh(
        Y, X, _contrail_values(cost_matrix),
        shading="nearest",
        cmap=cmap,
        norm=_contrail_norm(float(cost_matrix.min()), float(cost_matrix.max())),
        alpha=alpha
    )

    if colorbar:
        cbar = plt.colorbar(pcm, ax=ax_alt)
        cbar.set_label("Contrail cost")
        _contrail_cbar_format(cbar)
    return pcm


def _draw_contrail_map(fig, ax_map, results, trajectories, colors, windfield, barb_steps,
                       df_contrails, hour, bins, cmap, ts_select, real_df, real_label,
                       show_legend, colorbar=True, alpha=0.7):
    """Map panel of the *_with_contrails plots: contrail snapshot, tracks, wind, real flight.

    Returns the contrail contourf (None if there is none); ``colorbar=False`` leaves
    the colorbar to the caller.
    """
    _ctf_ref = None
    # ---- Contrail background (optional) ----
    if df_contrails is not None:
        dc = df_contrails.copy()
        # Select the single grid snapshot NEAREST to the requested time rather
        # than requiring an exact match: df_cost 'ts' is relative to departure
        # (rarely a whole hour), so an exact 3600*hour match usually finds nothing.
        if np.issubdtype(dc["ts"].dtype, np.number):
            grid_ts = np.sort(dc["ts"].unique())
            target = ts_select if ts_select is not None else 3600 * int(hour)
            chosen_ts = grid_ts[np.argmin(np.abs(grid_ts - target))]
            time_mask = dc["ts"] == chosen_ts
            ts_str = f"ts = {chosen_ts:.0f} s"
        else:
            dc["ts"] = pd.to_datetime(dc["ts"])
            hod = dc["ts"].dt.hour + dc["ts"].dt.minute / 60.0
            chosen_hour = int(dc["ts"].dt.hour.iloc[(hod - int(hour)).abs().values.argmin()])
            time_mask = dc["ts"].dt.hour == chosen_hour
            ts_str = f"{chosen_hour:02d}h UTC"

        visible_costs = dc.loc[time_mask, "cost"]
        vmin = float(visible_costs.min()) if len(visible_costs) > 0 else 0.0
        vmax = float(visible_costs.max()) if len(visible_costs) > 0 else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

        # Title carries the snapshot's UTC time when known (else the relative ts)
        # plus the shown flight level — instead of annotating inside the axes.
        map_title = ts_str
        if "timestamp" in dc.columns and time_mask.any():
            _utc = pd.Timestamp(dc.loc[time_mask, "timestamp"].iloc[0])
            map_title = _utc.strftime('%Y-%m-%d %H:%M UTC')

        top_alt = (
            dc.loc[time_mask & (dc["cost"] > 0)]
              .groupby("altitude")["cost"]
              .sum()
              .sort_values(ascending=False)
        )
        if len(top_alt) > 0:
            best_alt = float(top_alt.index[0])
            subset = dc[dc["altitude"] == best_alt][["latitude", "longitude", "cost"]].dropna()
            df_pivot = (subset
                        .pivot_table(index="latitude", columns="longitude", values="cost", aggfunc="mean")
                        .sort_index()
                        .sort_index(axis=1))
            lat_g = df_pivot.index.values
            lon_g = df_pivot.columns.values
            val = df_pivot.values
            Lon, Lat = np.meshgrid(lon_g, lat_g)
            if CONTRAIL_LOG_FLOOR is not None:
                # Log-spaced levels: linear ones would put every residual in the first bin.
                norm = _contrail_norm(vmin, vmax)
                levels = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), int(bins) + 1)
                ctf = ax_map.contourf(Lon, Lat, _contrail_values(val), levels=levels,
                                      norm=norm, cmap=cmap, alpha=alpha,
                                      transform=ccrs.PlateCarree())
            else:
                ctf = ax_map.contourf(Lon, Lat, val, levels=bins, cmap=cmap, alpha=alpha,
                                      transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
            fl = int(round(best_alt / 100.0))
            map_title += f"  |  FL{fl}"
            _ctf_ref = ctf

        # The body size of the current rc (10 by default, as before): a caller drawing
        # in a larger rc_context gets this title at its size too.
        ax_map.set_title(map_title, fontsize=plt.rcParams["font.size"])

    # Map features
    ax_map.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax_map.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax_map.add_feature(BORDERS, lw=0.5, color="gray")
    gl = ax_map.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    # cartopy makes these labels at DRAW time, so they would miss any rc_context the
    # caller drew the figure in: fix their font and size now, from the current rc.
    gl.xlabel_style = gl.ylabel_style = {"size": plt.rcParams["xtick.labelsize"],
                                         "family": plt.rcParams["font.family"]}
    gl.right_labels = False
    gl.top_labels = False
    ax_map.coastlines(resolution="50m", lw=0.5, color="gray")

    for i, df in enumerate(trajectories):
        # Plot trajectory
        if results[i].get("coef") is not None:
            ax_map.plot(df.longitude, df.latitude, color=colors[i], linewidth=2, marker='.', transform=ccrs.Geodetic(),
                        label=f'{results[i]["coef"]} + wind' if is_number(results[i]["coef"]) else results[i]["coef"]
                        )
        elif results[i].get("epsilon_factor") is not None:
            ax_map.plot(df.longitude, df.latitude, color=colors[i], linewidth=2, marker='.', transform=ccrs.Geodetic(),
                        label=f'factor {results[i]["epsilon_factor"]}')
        else:
            print("Error")
        # Plot start/end
        ax_map.scatter(df.longitude.iloc[0], df.latitude.iloc[0], c=colors[i], marker='o', transform=ccrs.Geodetic())
        ax_map.scatter(df.longitude.iloc[-1], df.latitude.iloc[-1], c=colors[i], marker='X', transform=ccrs.Geodetic())

        # Optional wind
        if windfield is not None:
            h_median = df.altitude.median()
            idx = np.argmin(abs(windfield.h.unique() - h_median))
            df_wind = windfield.query(f"h=={windfield.h.unique()[idx]}")
            ax_map.barbs(
                df_wind.longitude.values[::barb_steps],
                df_wind.latitude.values[::barb_steps],
                df_wind.u.values[::barb_steps],
                df_wind.v.values[::barb_steps],
                transform=ccrs.PlateCarree(),
                color="black",
                length=5,
                lw=0.5
            )

    if real_df is not None:
        ax_map.plot(real_df["longitude"], real_df["latitude"], color="black", ls="--",
                    lw=2, transform=ccrs.Geodetic(), zorder=6, label=real_label)
        ax_map.scatter(real_df["longitude"].iloc[0], real_df["latitude"].iloc[0],
                       c="black", marker="o", transform=ccrs.Geodetic(), zorder=6)
        ax_map.scatter(real_df["longitude"].iloc[-1], real_df["latitude"].iloc[-1],
                       c="black", marker="X", transform=ccrs.Geodetic(), zorder=6)

    if _ctf_ref is not None and colorbar:
        cbar = fig.colorbar(_ctf_ref, ax=ax_map, orientation="vertical", shrink=0.8, pad=0.02)
        cbar.set_label("Contrail cost", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if show_legend:
        ax_map.legend()
    return _ctf_ref


def plot_multiple_trajectories_with_contrails(results=None, windfield=None, barb_steps=10, filter_fuel_max=None,
                               df_contrails=None, hour=3, bins=12, cmap="Reds",
                               ts_select=None, alt_min=29000, alt_max=45000,
                               real_trajectory=None, real_label="Real flight",
                               show_legend=True):
    """
    Plot multiple trajectories with altitude, TAS, VS, and map.

    trajectories: list of DataFrames with ['ts','altitude','tas','vertical_rate','latitude','longitude']
    windfield: optional windfield DataFrame
    df_contrails: optional DataFrame with ['ts','cost','latitude','longitude','altitude'] for background
    hour: hour index used to select the contrail snapshot (when df_contrails is provided)
    bins: number of contour levels for the contrail background
    cmap: colormap for the contrail background
    real_trajectory: optional real flown trajectory to overlay (black dashed).
        A FlightData instance, a traffic Flight/Traffic, or a DataFrame with
        latitude/longitude/altitude (+ vertical_rate/groundspeed/ts if present).
    real_label: legend label for the real trajectory.
    show_legend: draw the legends (time-series panels + map). Set False for a
        clean figure, e.g. when the tracks are described in a caption instead.
    """
    (results, windfield, trajectories, colors, traj_dists,
     real_df, real_dist, ref_df) = _prepare_trajectories_for_contrail_plot(
        results, windfield, filter_fuel_max, real_trajectory)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(3, 2, figure=fig)

    # Time-series plots
    ax_alt = fig.add_subplot(gs[0, 0])
    ax_tas = fig.add_subplot(gs[1, 0])
    ax_vs = fig.add_subplot(gs[2, 0])

    if df_contrails is not None:
        _draw_altitude_contrail_background(ax_alt, df_contrails, ref_df, ts_select, alt_min, alt_max, cmap)

    for i, df in enumerate(trajectories):
        x_alt = traj_dists[i] if df_contrails is not None else df["ts"]
        ax_alt.plot(x_alt, df.altitude, lw=2, marker='.', color=colors[i])
        ax_tas.plot(df.ts, df.tas, lw=2, marker='.', color=colors[i])
        ax_vs.plot(df.ts, df.vertical_rate, lw=2, marker='.', color=colors[i])

    if real_df is not None:
        x_alt_real = real_dist if df_contrails is not None else real_df["ts"]
        ax_alt.plot(x_alt_real, real_df["altitude"], lw=2, ls="--",
                    color="black", zorder=6, label=real_label)
        if "tas" in real_df.columns:
            ax_tas.plot(real_df["ts"], real_df["tas"], lw=2, ls="--",
                        color="black", zorder=6, label=real_label)
        if "vertical_rate" in real_df.columns:
            ax_vs.plot(real_df["ts"], real_df["vertical_rate"], lw=2, ls="--",
                       color="black", zorder=6, label=real_label)

    ax_alt.set_ylabel("Altitude (ft)")
    ax_alt.set_xlabel("Distance (km)" if df_contrails is not None else "Time (s)")
    ax_alt.grid(ls=':')
    ax_tas.set_ylabel("TAS (kt)")
    ax_tas.grid(ls=':')
    ax_vs.set_ylabel("VS (ft/min)")
    ax_vs.set_xlabel("Time (s)")
    ax_vs.set_ylim(-3000, 3000)
    ax_vs.grid(ls=':')

    if show_legend:
        ax_alt.legend()
        ax_tas.legend()
        ax_vs.legend()

    # Map plot
    ax_map = fig.add_subplot(gs[:, 1],
                             projection=ccrs.TransverseMercator(
                                 central_longitude=ref_df.longitude.mean(),
                                 central_latitude=ref_df.latitude.mean()))
    _draw_contrail_map(fig, ax_map, results, trajectories, colors, windfield, barb_steps,
                       df_contrails, hour, bins, cmap, ts_select, real_df, real_label,
                       show_legend)
    plt.tight_layout()


def plot_altitude_2d_map_with_contrails(results=None, windfield=None, barb_steps=10, filter_fuel_max=None,
                                        df_contrails=None, hour=3, bins=12, cmap="Reds",
                                        ts_select=None, alt_min=29000, alt_max=45000,
                                        real_trajectory=None, real_label="Real flight",
                                        show_legend=True, bg_alpha=0.7, map_legend=True,
                                        x_axis="distance", bg_follow_time=True):
    """
    Same as plot_multiple_trajectories_with_contrails, but only the altitude
    profile (left) and the 2D map (right), without the TAS / VS panels.

    Takes the same arguments as plot_multiple_trajectories_with_contrails.
    Both panels share ONE colour scale and one colorbar (on the right), so the
    same colour means the same contrail cost in the profile and on the map.
    bg_alpha: opacity of the contrail background, the SAME on both panels and on
        the colorbar (otherwise one cost would look lighter on one panel).
    map_legend: draw the legend on the map too. The profile's legend already names
        every line, so False drops the duplicate.
    x_axis: "distance" (km along each track) or "time" (minutes since departure, from
        each trajectory's ``ts``). With "time" the contrail background is placed by
        the FIRST trajectory's timing.
    bg_follow_time: the profile's background is the cost along the FIRST trajectory's
        track at the time it is there (its ``ts``, linear between grid hours). The
        other trajectories pass the same ground a few minutes earlier or later, so for
        them it is the field up to that shift. False = one grid hour, the one nearest
        ``ts_select``. The map panel is always that one hour: a 2D map has no single time.
    """
    if x_axis not in ("distance", "time"):
        raise ValueError(f"x_axis={x_axis!r}; expected 'distance' or 'time'")
    (results, windfield, trajectories, colors, traj_dists,
     real_df, real_dist, ref_df) = _prepare_trajectories_for_contrail_plot(
        results, windfield, filter_fuel_max, real_trajectory)
    by_time = x_axis == "time"

    fig = plt.figure(figsize=FIGSIZE["alt_map"])
    # Two columns, close together. The colorbar is NOT a third GridSpec column: the
    # map keeps its geographic aspect, so it never fills its cell, and a separate
    # column left a wide gap between map and colorbar. It is an inset of the map
    # instead (below), so it follows the map's real frame.
    gs = GridSpec(1, 2, figure=fig, wspace=0.30)   # room for the map's latitude labels

    # Altitude profile
    ax_alt = fig.add_subplot(gs[0, 0])

    pcm = None
    if df_contrails is not None:
        pcm = _draw_altitude_contrail_background(
            ax_alt, df_contrails, ref_df, ts_select, alt_min, alt_max, cmap,
            colorbar=False, alpha=bg_alpha,
            x_ref=ref_df["ts"].values / 60.0 if by_time else None,
            follow_time=bg_follow_time)

    for i, df in enumerate(trajectories):
        if by_time:
            x_alt = df["ts"].values / 60.0
        else:
            x_alt = traj_dists[i] if df_contrails is not None else df["ts"]
        lbl = results[i].get("coef")
        ax_alt.plot(x_alt, df.altitude, lw=2, marker='.', color=colors[i],
                    label=lbl if isinstance(lbl, str) else None)

    if real_df is not None:
        if by_time:
            x_alt_real = real_df["ts"].values / 60.0
        else:
            x_alt_real = real_dist if df_contrails is not None else real_df["ts"]
        ax_alt.plot(x_alt_real, real_df["altitude"], lw=2, ls="--",
                    color="black", zorder=6, label=real_label)

    ax_alt.set_ylabel("Altitude (ft)")
    if by_time:
        ax_alt.set_xlabel("Time since departure (min)")
    else:
        ax_alt.set_xlabel("Distance (km)" if df_contrails is not None else "Time (s)")
    ax_alt.grid(ls=':')

    if show_legend:
        ax_alt.legend()

    # Map plot
    ax_map = fig.add_subplot(gs[0, 1],
                             projection=ccrs.TransverseMercator(
                                 central_longitude=ref_df.longitude.mean(),
                                 central_latitude=ref_df.latitude.mean()))
    ctf = _draw_contrail_map(fig, ax_map, results, trajectories, colors, windfield, barb_steps,
                             df_contrails, hour, bins, cmap, ts_select, real_df, real_label,
                             show_legend and map_legend, colorbar=False, alpha=bg_alpha)
    # Hug the profile: the map's spare width (its aspect is fixed) goes to the right,
    # where bbox_inches="tight" trims it, not between the two panels.
    ax_map.set_anchor("W")

    # One scale for both panels: the union of their ranges. set_clim re-colours the
    # contourf too (its levels stay where they were, only the colour mapping changes).
    shown = [m for m in (pcm, ctf) if m is not None]
    if shown:
        vmin = min(float(np.nanmin(m.get_array())) for m in shown)
        vmax = max(float(np.nanmax(m.get_array())) for m in shown)
        if vmin == vmax:
            vmax = vmin + 1e-6
        if CONTRAIL_LOG_FLOOR is not None:
            vmin = CONTRAIL_LOG_FLOOR   # the masked cells are not in the range
        for m in shown:
            m.set_clim(vmin, vmax)
        # Inset in axes coordinates: 1.5 % of the map's width to its right, whatever
        # width the aspect leaves it.
        cbar = fig.colorbar(shown[0], cax=ax_map.inset_axes([1.015, 0.0, 0.025, 1.0]))
        cbar.set_label("Contrail cost")
        _contrail_cbar_format(cbar)
    plt.tight_layout()

    # Shrink the map's column (and the figure with it) to the width the map really
    # takes: its aspect is geographic, so a half-width column leaves it floating in
    # empty space on both sides and pushes the colorbar away.
    fig.canvas.draw()
    cell_w = gs[0, 1].get_position(fig).width
    map_w = ax_map.get_position().width
    if 0 < map_w < cell_w:
        alt_w = ax_alt.get_position().width
        gs.set_width_ratios([alt_w, map_w])
        w, h = fig.get_size_inches()
        fig.set_size_inches(w - (cell_w - map_w) * w, h)
        plt.tight_layout()


def plot_altitude_vs_distance_all_trajectories(
        results, df_contrails=None, cmap="Reds",
        ts_select=None, alt_min=29000, alt_max=45000, save_dir=None,
        route=None, filename="altitude_vs_distance.png",
        real_trajectory=None, real_label="Real flight", field=None):
    """
    Contrail heatmap (linear scale, or log with CONTRAIL_LOG_FLOOR) in (distance, altitude) space with all
    trajectories overlaid.  alt_min/alt_max in feet — same unit as
    df_contrails['altitude'] and trajectory['altitude'].

    Accepts both opentop results (with 'trajectory' DataFrame) and graph-based
    results (with 'path' list of tuples). For graph-based results, pass the
    route from great_circle_route() via the `route` parameter to enable the
    contrail heatmap background.

    real_trajectory: optional real flown flight overlaid as a black dashed line
        (its altitude vs cumulative great-circle distance). A FlightData, a
        traffic Flight/Traffic, or a DataFrame with latitude/longitude/altitude.
    field: optional background to draw INSTEAD of re-sampling ``df_contrails`` at one
        hour: ``{"dist": (n,), "alt": (m,), "cost": (n, m), "label": str}``, e.g. the
        field a solver actually optimised over. ``df_contrails`` is then not read.
    """
    from matplotlib.ticker import FuncFormatter

    def haversine_cum(lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        a = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
        return np.concatenate([[0.0], np.cumsum(6371.0 * 2 * np.arcsin(np.sqrt(a)))])

    is_graph = results and "trajectory" not in results[0] and (
        "path" in results[0] or ("alt_ft" in results[0] and "dist_km" in results[0])
    )

    if is_graph:
        if route is not None:
            ref_lats  = np.array([p[0] for p in route])
            ref_lons  = np.array([p[1] for p in route])
            ref_dists = np.array([p[2] for p in route])
        else:
            # no route provided — build x-axis from first result
            first = results[0]
            if "path" in first:
                ref_dists = np.array([p[0] for p in first["path"]])
            else:
                ref_dists = np.asarray(first["dist_km"])
            ref_lats = ref_lons = None
        total_dist = float(ref_dists[-1])
    else:
        ref_traj  = results[0]["trajectory"]
        ref_lats  = ref_traj["latitude"].values
        ref_lons  = ref_traj["longitude"].values
        ref_dists = haversine_cum(ref_lats, ref_lons)
        total_dist = float(ref_dists[-1])

    cost_matrix = None
    alt_levels  = None
    ts_label    = ""

    if field is not None:
        ref_dists   = np.asarray(field["dist"], dtype=float)
        alt_levels  = np.asarray(field["alt"], dtype=float)
        cost_matrix = np.asarray(field["cost"], dtype=float)
        ts_label    = field.get("label", "")
    elif df_contrails is not None:
        dfc      = df_contrails.copy()
        grid_ts  = np.sort(dfc["ts"].unique())
        chosen_ts = (grid_ts[np.argmin(np.abs(grid_ts - ts_select))]
                     if ts_select is not None else grid_ts[len(grid_ts) // 2])

        if "timestamp" in dfc.columns:
            mask = dfc["ts"] == chosen_ts
            if mask.any():
                ts_label = pd.Timestamp(dfc.loc[mask, "timestamp"].iloc[0]).strftime("%Hh%M UTC")

        lo, hi = _enclosing_alt_levels(dfc, alt_min, alt_max)
        dfc_slice = (
            dfc[(dfc["ts"] >= chosen_ts - 1800) & (dfc["ts"] <= chosen_ts + 1800) &
                (dfc["altitude"] >= lo) & (dfc["altitude"] <= hi)]
            .groupby(["latitude", "longitude", "altitude"], as_index=False)
            .agg({"cost": "max"})
        )

        alt_levels = np.array(sorted(dfc_slice["altitude"].unique()), dtype=float)
        print(f"Contrail slice: {len(alt_levels)} alt levels "
              f"({alt_levels.min() if len(alt_levels) else 0:.0f}–"
              f"{alt_levels.max() if len(alt_levels) else 0:.0f} ft) | ts={ts_label}")

        if len(alt_levels) >= 2 and ref_lats is not None and ref_lons is not None:
            cost_matrix = np.zeros((len(ref_dists), len(alt_levels)))
            for j, alt in enumerate(alt_levels):
                layer = dfc_slice[dfc_slice["altitude"] == alt]
                if layer.empty:
                    continue
                lats_g = np.sort(layer["latitude"].unique())
                lons_g = np.sort(layer["longitude"].unique())
                if len(lats_g) < 2 or len(lons_g) < 2:
                    continue
                pivot = (layer.pivot_table(index="latitude", columns="longitude",
                                           values="cost", aggfunc="mean")
                         .reindex(index=lats_g, columns=lons_g).fillna(0))
                interp_layer = RegularGridInterpolator(
                    (lats_g, lons_g), pivot.values,
                    method="linear", bounds_error=False, fill_value=0.0)
                cost_matrix[:, j] = np.maximum(
                    0, interp_layer(np.column_stack([ref_lats, ref_lons])))

    fig, ax = plt.subplots(figsize=FIGSIZE["altitude"])

    if cost_matrix is not None and cost_matrix.max() > 0:
        X, Y = np.meshgrid(alt_levels, ref_dists)
        # "nearest": each value fills the cell around its node -- for the DP's cost_map
        # that IS the field it optimises (get_contrail_cost rounds to the nearest node).
        pcm = ax.pcolormesh(Y, X, _contrail_values(cost_matrix), shading="nearest", cmap=cmap,
                            norm=_contrail_norm(0.0, float(cost_matrix.max())))
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label("Contrail cost")
        _contrail_cbar_format(cbar)

    if is_graph:
        # Also accept DPPareto results that store alt_ft / dist_km instead of path.
        def _get_dist_alt(result):
            if "path" in result:
                path = result["path"]
                return np.array([p[0] for p in path]), np.array([p[1] for p in path])
            return np.asarray(result["dist_km"]), np.asarray(result["alt_ft"])

        # Decide legend mode: coef-per-line when all results share one optimizer.
        optimizers = list(dict.fromkeys(
            r.get("optimizer", r.get("method", "unknown")) for r in results
        ))
        single_optimizer = len(optimizers) == 1

        if single_optimizer:
            # One color per coef value using a sequential colormap.
            # DP_MuO has no coef: it returns a FRONT, not a scalarised
            # sweep. Fall back to the point's index along the front so the ramp
            # still separates the curves instead of drawing them all one colour.
            coefs      = [float(r.get("coef") or r.get("point") or 0) for r in results]
            coef_min   = min(coefs)
            coef_max   = max(coefs)
            norm_coef  = plt.Normalize(coef_min, coef_max) if coef_max > coef_min else plt.Normalize(0, 1)
            cmap_traj  = cm.plasma

            for result, coef in zip(results, coefs):
                dists, alts = _get_dist_alt(result)
                color = cmap_traj(norm_coef(coef))
                eps = result.get("epsilon_factor") or result.get("fuel_slack_frac")
                if eps is not None:
                    label = f"ε ≤ {eps:.2f}"
                elif result.get("point") is not None:
                    label = f"Pareto #{int(result['point'])}"
                else:
                    label = f"coef = {coef:.2f}"
                ax.plot(dists, alts, lw=1.5, color=color, label=label, zorder=4)
        else:
            # Multiple optimizers: one color per optimizer, alpha encodes coef.
            colors    = cm.tab10(np.linspace(0, 1, max(len(optimizers), 1)))
            color_map = {m: colors[i] for i, m in enumerate(optimizers)}
            seen      = set()
            for result in results:
                dists, alts = _get_dist_alt(result)
                opt   = result.get("optimizer", result.get("method", "unknown"))
                coef  = result.get("coef", 0) or 0
                label = opt if opt not in seen else "_nolegend_"
                seen.add(opt)
                alpha = 0.4 + 0.6 * float(coef)
                ax.plot(dists, alts, lw=1.5, color=color_map[opt],
                        alpha=alpha, label=label, zorder=4)
    else:
        traj_colors = cm.tab20(range(len(results) + 1))
        has_failed = False
        for i, result in enumerate(results):
            traj = result["trajectory"].copy()
            traj["dist"] = haversine_cum(traj["latitude"].values, traj["longitude"].values)
            # Legend shows the fuel-mass multiplying factor (ε), as in the DP branch,
            # not the absolute fuel budget (fuel_max / total mass).
            eps = result.get("epsilon_factor")
            if eps is None:
                eps = result.get("fuel_slack_frac")
            if isinstance(eps, (int, float)):
                label = f"ε < {eps:.2f}"
            elif eps is not None:
                label = str(eps)  # e.g. "great_circle" baseline
            else:
                coef = result.get("coef")
                label = f"coef = {coef:.2f}" if isinstance(coef, (int, float)) else f"traj {i}"
            converged = result.get("solver_success", True)
            color = traj_colors[i] if converged else "black"
            if not converged:
                has_failed = True
            ax.plot(traj["dist"], traj["altitude"], lw=2, marker=".",
                    color=color, label=label, zorder=4)
        if has_failed:
            ax.plot([], [], color="black", lw=2, marker=".", label="not converged")

    # Optional real flown trajectory (black dashed): altitude vs cumulative
    # great-circle distance, on the same axes as the optimized profiles.
    if real_trajectory is not None:
        real_df = _real_trajectory_df(real_trajectory)
        real_dist = haversine_cum(real_df["latitude"].values, real_df["longitude"].values)
        ax.plot(real_dist, real_df["altitude"], lw=2, ls="--", color="black",
                zorder=6, label=real_label)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (ft)")

    # ── Title: solver | flight | Mach (like plot_benchmarking_results) ──
    first   = results[0]
    solver  = first.get("optimizer") or first.get("method")
    if solver is None:
        solver = "DP" if is_graph else "OpenTOP"
    flight  = first.get("flight_name") or first.get("Flight")
    mach    = first.get("mach") or first.get("INPUT_mach")
    if mach is None and not is_graph:
        rt = first.get("trajectory")
        if rt is not None and "mach" in getattr(rt, "columns", []):
            mach = float(rt["mach"].iloc[0])
    title_parts = [str(solver)]
    if flight:
        title_parts.append(str(flight))
    if mach is not None:
        title_parts.append(f"Mach {float(mach):.3f}")
    title = "Trajectories — " + "  |  ".join(title_parts)
    if ts_label:
        title += f"  |  {ts_label}"
    ax.set_title(title)
    ax.set_xlim(0, total_dist)
    ax.set_ylim(alt_min, alt_max)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"FL{int(v) // 100}"))
    ax.legend(loc="upper right", fontsize=_pt(8), frameon=True, draggable=True)
    ax.grid(ls=":", alpha=0.5)
    plt.tight_layout()
    if save_dir:
        out_path = os.path.join(save_dir, filename)
        if os.path.exists(out_path):
            os.remove(out_path)
        plt.savefig(out_path, dpi=300)
        print(f"Saved altitude plot to {out_path}")
    return fig

# Benchmarking plots ------------------------------------------------------------------------------------------------------------------

def plot_benchmarking_results(dp, opentop, save_dir=None, alt_min_ft=None, alt_max_ft=None, filename="benchmarking.png"):
    # Accept dp as a list of dicts (one per coef) or a single dict
    if isinstance(dp, dict):
        dp_list = [dp]
    else:
        dp_list = list(dp)

    dp0    = dp_list[0]
    MACH   = float(dp0["INPUT_mach"])
    MASS   = float(dp0["INPUT_mass_init_kg"])
    ACTYPE = dp0["actype"]

    env_distances = np.array(dp0["env_distances"])
    env_altitudes = np.array(dp0["env_altitudes"])
    env_cost_map  = np.array(dp0["env_cost_map"])
    env_dist_step = float(dp0["env_dist_step_km"])

    # ── Cost map lookup (distance-indexed, shared by DP and OpenTOP) ──────
    def get_cost(d, alt):
        di = int(np.clip(np.round(d / env_dist_step), 0, len(env_distances) - 1))
        da = env_altitudes[1] - env_altitudes[0]
        ai = int(np.clip(np.round((alt - env_altitudes[0]) / da), 0, len(env_altitudes) - 1))
        return float(env_cost_map[di, ai])

    # ── DP cumulative fuel (per-edge, with real mass depletion) ───────────
    def compute_cum_fuel(alt_ft, ts_s, mach, mass_init_kg, actype):
        ff_model = FuelFlow(actype, wave_drag=True, use_synonym=True, force_engine=True)
        cum  = [0.0]
        mass = mass_init_kg
        for i in range(len(alt_ft) - 1):
            avg_alt = (alt_ft[i] + alt_ft[i + 1]) / 2.0
            dt  = ts_s[i + 1] - ts_s[i]
            vs  = (alt_ft[i + 1] - alt_ft[i]) / (dt / 60.0)
            tas = aero.mach2tas(mach, avg_alt * ft) / kts
            ff  = ff_model.enroute(mass=mass, tas=tas, alt=avg_alt, vs=vs)
            fuel = max(0.0, ff) * dt
            cum.append(cum[-1] + fuel)
            mass -= fuel
        return np.array(cum)

    # ── Build processed DP trajectory list ────────────────────────────────
    dp_trajs = []
    for rec in dp_list:
        dp_alt  = np.array(rec["alt_ft"])
        dp_dist = np.array(rec["dist_km"])
        dp_ts   = np.array(rec["ts_s"])
        coef    = rec.get("Coef", rec.get("coef", 0.0))

        fuel_cum = compute_cum_fuel(dp_alt, dp_ts, MACH, MASS, ACTYPE)

        con_cum = [0.0]
        for i in range(1, len(dp_dist)):
            avg_d = (dp_dist[i - 1] + dp_dist[i]) / 2.0
            avg_a = (dp_alt[i - 1]  + dp_alt[i])  / 2.0
            dt_s  = dp_ts[i] - dp_ts[i - 1]
            con_cum.append(con_cum[-1] + get_cost(avg_d, avg_a) * dt_s)

        dp_trajs.append({
            "alt":             dp_alt,
            "dist":            dp_dist,
            "ts_min":          dp_ts / 60.0,
            "fuel_cum":        fuel_cum,
            "con_cum":         np.array(con_cum),
            "coef":            coef,
            "fuel_slack_frac": rec.get("fuel_slack_frac"),
            "metrics":         rec.get("metrics", {}),
            "time":            rec.get("time", 0.0),
        })

    def extract_traj(traj):
        lats = traj["latitude"].values
        lons = traj["longitude"].values
        alts = traj["altitude"].values
        fuel = traj["fuel_cost"].fillna(0).values

        dist = [0.0]
        for i in range(1, len(lats)):
            d = aero.distance(lats[i - 1], lons[i - 1], lats[i], lons[i]) / 1000
            dist.append(dist[-1] + d)
        dist = np.array(dist)

        if "time" in traj.columns:
            ts_min = (traj["time"].values - traj["time"].values[0]) / 60.0
        else:
            cruise_speed_kmpm = aero.mach2tas(MACH, 35000 * ft) * 3.6 / 60
            ts_min = dist / cruise_speed_kmpm

        fuel_cum = np.cumsum(np.concatenate([[0], fuel[:-1]]))

        con_cum = [0.0]
        for i in range(1, len(dist)):
            avg_d = (dist[i - 1] + dist[i]) / 2.0
            avg_a = (alts[i - 1] + alts[i]) / 2.0
            dt_s  = (ts_min[i] - ts_min[i - 1]) * 60.0
            con_cum.append(con_cum[-1] + get_cost(avg_d, avg_a) * dt_s)

        return {
            "alt":        np.array(alts),
            "dist":       dist,
            "ts_min":     ts_min,
            "fuel_cum":   fuel_cum,
            "con_cum":    np.array(con_cum),
            "fuel_total": float(fuel_cum[-1]),
            "con_total":  float(con_cum[-1]),
        }

    # ── Classify OpenTOP trajectories ─────────────────────────────────────
    baseline   = None
    baseline_r = None
    contrails  = []
    opentop_r  = None

    for r in opentop:
        if r["trajectory"] is None:
            continue
        t = extract_traj(r["trajectory"])
        # Fuel-optimal baseline: the weighted-sum "great circle" point, or the
        # ε-constraint point at ε ≤ 1.0 (budget == fuel-optimal). Both are drawn
        # as the black dashed "OpenTOP fuel-opt" reference line.
        _frac = r.get("fuel_slack_frac")
        if (r["coef"] == "great circle" or r.get("fuel_max") == "great circle"
                or (_frac is not None and _frac <= 1.0)):
            baseline   = t
            baseline_r = r
            continue
        # Legend shows the fuel-mass multiplying factor (ε), as in the DP branch,
        # not the absolute fuel budget (fuel_max / total mass).
        frac = r.get("fuel_slack_frac")
        if frac is None:
            frac = r.get("epsilon_factor")
        if isinstance(frac, (int, float)):
            label = f"ε ≤ {frac:.2f}"
        elif frac is not None:
            label = str(frac)
        else:
            label = f"coef={r['coef']}"
        t["label"] = label
        t["coef"]  = r["coef"]                       # match key for weighted sum
        t["fuel_slack_frac"] = r.get("fuel_slack_frac")  # match key for ε-constraint
        contrails.append(t)
        opentop_r = r

    # lookups → OpenTOP final fuel / contrail value (for DP legend labels).
    # Weighted-sum runs are matched by coef; ε-constraint runs (coef is None for
    # both solvers) are matched by the fuel-mass multiplying factor instead.
    ot_fuel_by_coef = {ct["coef"]: ct["fuel_cum"][-1] for ct in contrails if ct.get("coef") is not None}
    ot_con_by_coef  = {ct["coef"]: ct["con_cum"][-1]  for ct in contrails if ct.get("coef") is not None}
    ot_fuel_by_frac = {ct["fuel_slack_frac"]: ct["fuel_cum"][-1] for ct in contrails if ct.get("fuel_slack_frac") is not None}
    ot_con_by_frac  = {ct["fuel_slack_frac"]: ct["con_cum"][-1]  for ct in contrails if ct.get("fuel_slack_frac") is not None}
    if baseline is not None:
        ot_fuel_by_coef[0.0] = baseline["fuel_cum"][-1]
        ot_con_by_coef[0.0]  = baseline["con_cum"][-1]

    def _ot_ref(dt, by_frac, by_coef):
        """Matching OpenTOP value and a reference tag for this DP trajectory.

        ε-constraint trajectories carry a fuel_slack_frac and are matched on it;
        weighted-sum trajectories fall back to coef matching.
        """
        frac = dt.get("fuel_slack_frac")
        if frac is not None:
            v = by_frac.get(frac)
            return (v, f"OT ε ≤ {frac:.2f}") if v is not None else (None, None)
        coef = dt["coef"]
        v = by_coef.get(coef)
        if v is None:
            return None, None
        return v, ("OT fuel-opt" if coef == 0.0 else f"OT {coef}")

    # ── Print summary ─────────────────────────────────────────────────────
    if baseline_r is not None:
        b = baseline
        print(f"OpenTOP baseline:     fuel={b['fuel_total']:.1f} kg | contrail={b['con_total']:.4f}"
              f" | time={baseline_r.get('opentop_time_s', float('nan')):.2f} s")
    if opentop_r is not None:
        ot = contrails[-1]
        print(f"OpenTOP coef={opentop_r['coef']}: fuel={ot['fuel_total']:.1f} kg"
              f" | fuel_cost={opentop_r['fuel_cost']:.4f} | grid_cost={opentop_r['grid_cost']:.4f}"
              f" | time={opentop_r.get('opentop_time_s', float('nan')):.2f} s")
    for dt in dp_trajs:
        m = dt["metrics"]
        print(f"DP      coef={dt['coef']}: fuel={m.get('total_fuel_kg', 0.0):.1f} kg"
              f" | contrail={m.get('total_contrail', 0.0):.4f}"
              f" | time={dt['time']:.2f} s")

    # ── Reference: OpenTOP fuel-opt baseline (falls back to DP coef=0) ───
    if baseline is not None:
        ref_fuel = baseline["fuel_cum"][-1]
        ref_con  = baseline["con_cum"][-1]
        ref_tag_default = "OT fuel-opt"
    else:
        _dp_ref  = next((dt for dt in dp_trajs if dt["coef"] == 0.0), dp_trajs[0])
        ref_fuel = _dp_ref["fuel_cum"][-1]
        ref_con  = _dp_ref["con_cum"][-1]
        ref_tag_default = "DP fuel-opt"

    def _norm_fuel(arr):
        return arr / ref_fuel * 100.0 if ref_fuel else arr

    def _norm_con(arr):
        return arr / ref_con * 100.0 if ref_con else arr

    def _pct_lbl(val, ref):
        return f"{val / ref * 100:.1f}%" if ref else "n/a"

    # ── Plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(2, 2)

    # ── Trajectory on cost map ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    X, Y = np.meshgrid(env_distances, env_altitudes)
    pcm = ax.pcolormesh(X, Y, env_cost_map.T, cmap="Reds")
    plt.colorbar(pcm, ax=ax, label=f"Contrail persistence (sigma={opentop_r['smooth_factor']})")

    n_dp = len(dp_trajs)
    dp_colors = plt.cm.Greens(np.linspace(0.35, 1.0, max(n_dp, 1)))
    ot_colors = plt.cm.Blues(np.linspace(0.35, 1.0, max(len(contrails), 1)))

    def _is_fuel_opt(dt):
        # Fuel-optimal point: coef==0 for weighted sum, ε==1.0 for ε-constraint.
        frac = dt.get("fuel_slack_frac")
        if frac is not None:
            return abs(frac - 1.0) < 1e-9
        return dt["coef"] == 0.0

    def _dp_lbl(dt):
        if _is_fuel_opt(dt):
            return "DP fuel-opt"
        frac = dt.get("fuel_slack_frac")
        return f"DP ε ≤ {frac:.2f}" if frac is not None else f"DP coef={dt['coef']}"

    def _dp_ls(dt):
        return "--" if _is_fuel_opt(dt) else "-"

    def _dp_color(dt, i):
        return "black" if _is_fuel_opt(dt) else dp_colors[i]

    if baseline:
        ax.plot(baseline["dist"], baseline["alt"], "k--", lw=2, label="OpenTOP fuel-opt")
    for i, ct in enumerate(contrails):
        ax.plot(ct["dist"], ct["alt"], color=ot_colors[i], alpha=0.7,
                label=f"OpenTOP {ct['label']}")
    for i, dt in enumerate(dp_trajs):
        lbl = _dp_lbl(dt) if n_dp > 1 else "DP"
        ax.plot(dt["dist"], dt["alt"], lw=2.0, ls=_dp_ls(dt), color=_dp_color(dt, i), label=lbl)

    ax.set_title(f"Trajectories — {ACTYPE}  |  Mach {MACH:.3f}  |  mass {MASS:.0f} kg")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_xlim(env_distances[0], env_distances[-1])
    y_lo = alt_min_ft if alt_min_ft is not None else env_altitudes[0]  - 500
    y_hi = alt_max_ft if alt_max_ft is not None else env_altitudes[-1] + 500
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper right").set_draggable(True)

    # ── Cumulative fuel vs time  (% of OpenTOP fuel-opt) ─────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if baseline:
        ax2.plot(baseline["ts_min"], _norm_fuel(baseline["fuel_cum"]),
                 "k--", lw=2, label=f"OpenTOP fuel-opt ({ref_fuel:.0f} kg)")
    for i, ct in enumerate(contrails):
        d = (ct["fuel_cum"][-1] - ref_fuel) / ref_fuel * 100 if ref_fuel else 0
        ax2.plot(ct["ts_min"], _norm_fuel(ct["fuel_cum"]),
                 color=ot_colors[i], alpha=0.7,
                 label=f"OpenTOP {ct['label']} ({d:+.1f}%)")
    for i, dt in enumerate(dp_trajs):
        ot_f, ref_tag = _ot_ref(dt, ot_fuel_by_frac, ot_fuel_by_coef)
        if ot_f:
            d = (dt["fuel_cum"][-1] - ot_f) / ot_f * 100
            delta = f"{d:+.1f}% vs {ref_tag}"
        else:
            d = (dt["fuel_cum"][-1] - ref_fuel) / ref_fuel * 100 if ref_fuel else 0
            delta = f"{d:+.1f}% vs {ref_tag_default}"
        lbl = f"{_dp_lbl(dt)} ({delta})" if n_dp > 1 else f"DP ({delta})"
        ax2.plot(dt["ts_min"], _norm_fuel(dt["fuel_cum"]),
                 lw=2.0, ls=_dp_ls(dt), color=_dp_color(dt, i), label=lbl)
    ax2.axhline(100, color="k", lw=0.8, ls=":", alpha=0.5)
    ax2.set_title("Cumulative fuel burn")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Fuel burned (% of OpenTOP fuel-opt)")
    ax2.legend(loc="upper left").set_draggable(True)

    # ── Cumulative contrail vs time  (% of OpenTOP fuel-opt) ─────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if baseline:
        ax3.plot(baseline["ts_min"], _norm_con(baseline["con_cum"]),
                 "k--", lw=2, label=f"OpenTOP fuel-opt ({ref_con:.4f})")
    for i, ct in enumerate(contrails):
        d = (ct["con_cum"][-1] - ref_con) / ref_con * 100 if ref_con else 0
        ax3.plot(ct["ts_min"], _norm_con(ct["con_cum"]),
                 color=ot_colors[i], alpha=0.7,
                 label=f"OpenTOP {ct['label']} ({d:+.1f}%)")
    for i, dt in enumerate(dp_trajs):
        ot_c, ref_tag = _ot_ref(dt, ot_con_by_frac, ot_con_by_coef)
        if ot_c:
            d = (dt["con_cum"][-1] - ot_c) / ot_c * 100
            delta = f"{d:+.1f}% vs {ref_tag}"
        else:
            d = (dt["con_cum"][-1] - ref_con) / ref_con * 100 if ref_con else 0
            delta = f"{d:+.1f}% vs {ref_tag_default}"
        lbl = f"{_dp_lbl(dt)} ({delta})" if n_dp > 1 else f"DP ({delta})"
        ax3.plot(dt["ts_min"], _norm_con(dt["con_cum"]),
                 lw=2.0, ls=_dp_ls(dt), color=_dp_color(dt, i), label=lbl)
    ax3.axhline(100, color="k", lw=0.8, ls=":", alpha=0.5)
    ax3.set_title("Cumulative contrail exposure")
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Contrail exposure (% of OpenTOP fuel-opt)")
    ax3.legend(loc="upper left").set_draggable(True)

    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, filename)
        if os.path.exists(out_path):
            os.remove(out_path)
        plt.savefig(out_path, dpi=300)
        print(f"Saved → {out_path}")
    plt.show()




if __name__ == "__main__":
    run_dir = RUN_DIR / "benchmarking"
    # flight_name="Ireland-Libya"
    flight_name='Germany-Turkey'
    # begin="2022-02-20 00:00"
    # end="2022-02-20 03:29"
    begin = "2022-02-20 08:42"
    end = "2022-02-20 12:43"
    sigma=2
    # save_dir=Path("Ireland-Libya")
    # base_id="710107_18154_0"
    base_id='06a062_00477_0'
    DATA_PATH = str(DATA_DIR / "eu_flights_2022feb20_filter_resample_meteo.parquet.gz")
    GRID_PATH = str(DATA_DIR / "grid_era5_smoothed.parquet.gz")
    print(f"Preparing flight: {flight_name} | {begin} → {end} | σ={sigma}")
    # Prepares interpolant cache for flight
    TMP = Path(tempfile.gettempdir())
    filename = begin.replace(" ", "_").replace(":", "-")
    # Tag the cache by meteo source: cached_interpolant_from_dataframe does no
    # invalidation, so grid- and ERA5-derived interpolants must not share a key.
    from src.data import GridWeather  # local: avoids import cycle
    domain_meteo = GridWeather(GRID_PATH, sigmas=[sigma])
    src_tag = "grid" if domain_meteo is not None else "era5"
    interpolant_cache = os.environ.get(
        "INTERPOLANT_CACHE",
        str(TMP / f"opentop_interp_{src_tag}_{flight_name}_{filename}_sigma{sigma}.casadi"),
    )
    fd = FlightData(start=begin, stop=end, path=DATA_PATH, base_id=base_id)


    # Slice the flight's cost grid from the shared precomputed ERA5 grid
    # (no per-flight ERA5 fetch, no agg_conditions/prepare_grid recompute).
    df_cost = domain_meteo.slice_for_flight(
        fd.initial_latitude, fd.initial_longitude,
        fd.final_latitude, fd.final_longitude,
        start=fd.start, stop=fd.stop, sigma=sigma,
    )

    # df_cost already built from the shared grid above (df_cost_for_flight);
    # no per-flight ERA5 (fd.meteo is None on the grid path).

    # Plot the real flown trajectory over the contrail field
    plot_multiple_trajectories_with_contrails(real_trajectory=fd, df_contrails=df_cost, hour=3, alt_min=20000, alt_max=45000)
    plt.show()
    #
    # df = pd.DataFrame([{
    #     "success": r["solver_success"],
    # } for r in results])
