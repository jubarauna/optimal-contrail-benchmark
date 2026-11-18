import numpy as np
import pandas as pd
from casadi import DM


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FOR A*
# ──────────────────────────────────────────────────────────────────────────────

class ContrailEnvironmentFromGrid:
    def __init__(
        self,
        interpolant,
        grid_data: pd.DataFrame,
        route: tuple,
        alt_step_ft: float = 1000,
        min_alt_ft: float = 29000,
        max_alt_ft: float = 39000,
        ts_offset_s: float = 0.0,
    ):
        self.interpolant = interpolant
        self.grid_data = grid_data.copy()
        self.route, self.total_distance = route
        self.alt_step_ft = alt_step_ft
        self.min_alt_ft = min_alt_ft
        self.max_alt_ft = max_alt_ft
        self.altitudes = np.arange(min_alt_ft, max_alt_ft + alt_step_ft, alt_step_ft)
        self.distances = np.array([d for _, _, d, _ in self.route])
        self._ts_offset_s = ts_offset_s
        self.dist_step_km = self.distances[1] - self.distances[0] if len(self.distances) > 1 else 10.0
        self.ts = np.array([ts for _, _, _, ts in self.route]) + ts_offset_s
        print(f"Environment: {len(self.distances)} distance pts × {len(self.altitudes)} altitude levels")
        self._sample_along_route()

    def _sample_along_route(self):
        n_dist = len(self.distances)
        n_alt = len(self.altitudes)
        self.cost_map = np.zeros((n_dist, n_alt))


        ts_min = float(self.grid_data["ts"].min())
        ts_max = float(self.grid_data["ts"].max())


        for i, point in enumerate(self.route):
            lat, lon, _, ts = point
            ts_c = float(np.clip(ts + self._ts_offset_s, ts_min, ts_max))
            for j, alt in enumerate(self.altitudes):
                h_m = alt * 0.3048
                try:
                    val = self.interpolant(DM([lon, lat, h_m, ts_c])).full().flatten()[0]
                except Exception:
                    val = 0.0
                self.cost_map[i, j] = max(val, 0.0)

        print(f"Cost map: {self.cost_map.min():.3f} – {self.cost_map.max():.3f}")

    def get_contrail_cost(self, distance_km: float, altitude_ft: float) -> float:
        dist_idx = int(np.clip(np.round(distance_km / self.dist_step_km), 0, len(self.distances) - 1))
        alt_idx = int(np.clip(np.round((altitude_ft - self.min_alt_ft) / self.alt_step_ft), 0, len(self.altitudes) - 1))
        return float(self.cost_map[dist_idx, alt_idx])

    def get_contrail_cost_by_ts(self, ts_s: float, altitude_ft: float) -> float:
        """
        Look up contrail cost using wall-clock time instead of distance.

        Finds the nearest route point by timestamp, then reads cost_map at
        that index.  Because ts and distances are 1-to-1 for the nominal
        route the *values* match get_contrail_cost(), but the lookup is
        semantically indexed by when the aircraft is there, not where.
        """
        ts_idx = int(np.clip(
            np.searchsorted(self.ts, ts_s), 0, len(self.ts) - 1
        ))
        alt_idx = int(np.clip(
            np.round((altitude_ft - self.min_alt_ft) / self.alt_step_ft),
            0, len(self.altitudes) - 1
        ))
        return float(self.cost_map[ts_idx, alt_idx])
