import pandas as pd
from openap import aero
from scipy.ndimage import gaussian_filter


def prepare_grid(contrail_df: pd.DataFrame, sigma: float, t_reference=None) -> pd.DataFrame:
    """
    Build the cost grid for opentop and A*.

    t_reference: the ts=0 origin.
      - Pass fd.start (departure time) so that ts=0 means "aircraft departs".
        Both opentop (which evaluates at elapsed flight time) and A* then
        automatically read the correct contrail snapshot.
      - If None, ts=0 is the first grid timestamp (old behaviour — off by the
        departure offset relative to the grid start).
    """
    if t_reference is None:
        t_ref = contrail_df["timestamp"].iloc[0]
    else:
        t_ref = pd.Timestamp(t_reference)
        # Match timezone-awareness of the grid timestamps.
        grid_tz = contrail_df["timestamp"].dt.tz
        if grid_tz is None and t_ref.tzinfo is not None:
            t_ref = t_ref.tz_localize(None)
        elif grid_tz is not None and t_ref.tzinfo is None:
            t_ref = t_ref.tz_localize("UTC")

    df_cost = (
        contrail_df.assign(
            height=lambda x: x.altitude * aero.ft,
            cost=lambda x: x.persistent.astype(float),
            ts=lambda d: (d.timestamp - t_ref).dt.total_seconds(),
        )
        .sort_values(["ts", "height", "latitude", "longitude"])
    )
    cost = df_cost.cost.values.reshape(
        df_cost.ts.nunique(),
        df_cost.height.nunique(),
        df_cost.latitude.nunique(),
        df_cost.longitude.nunique(),
    )
    cost_ = gaussian_filter(cost, sigma=(0, sigma, sigma, sigma), mode="nearest")
    df_cost=df_cost.assign(cost=cost_.flatten()).fillna(0)

    return df_cost
