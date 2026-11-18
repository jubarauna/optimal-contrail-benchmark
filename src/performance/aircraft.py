"""Aircraft performance (OpenAP): the one place the models and fuel burn are built.

Every solver and every post-hoc measure gets its aircraft data and fuel/thrust/drag
models from here, so they all use the same OpenAP settings (synonyms, forced
default engine, wave drag)."""
import numpy as np
from openap import aero, prop, FuelFlow, Thrust, Drag
from openap.aero import ft, kts


def aircraft(actype: str) -> dict:
    """OpenAP aircraft properties (``mtow``, ``wing``, ``limits``, ...)."""
    return prop.aircraft(actype, use_synonym=True)


def fuel_flow_model(actype: str) -> FuelFlow:
    return FuelFlow(actype, wave_drag=True, use_synonym=True, force_engine=True)


def thrust_model(actype: str) -> Thrust:
    return Thrust(actype, use_synonym=True, force_engine=True)


def drag_model(actype: str) -> Drag:
    return Drag(actype, wave_drag=True, use_synonym=True, force_engine=True)


def _burn(alt_ft, ts_s, mach: float, mass_init_kg: float, actype: str):
    """Cumulative fuel (kg, one value per point) and final mass along a trajectory.

    Per segment, OpenAP's en-route fuel flow at the mid-segment altitude, the TAS of
    the fixed cruise Mach there, the segment's vertical speed and the mass left at its
    start, times the segment duration; mass is depleted segment by segment.
    """
    ff_model = fuel_flow_model(actype)
    cum = [0.0]
    mass = mass_init_kg
    for i in range(len(alt_ft) - 1):
        avg_alt = (alt_ft[i] + alt_ft[i + 1]) / 2.0
        dt = ts_s[i + 1] - ts_s[i]
        vs = (alt_ft[i + 1] - alt_ft[i]) / (dt / 60.0)
        tas = aero.mach2tas(mach, avg_alt * ft) / kts

        ff = ff_model.enroute(mass=mass, tas=tas, alt=avg_alt, vs=vs)
        fuel = max(0.0, ff) * dt
        cum.append(cum[-1] + fuel)
        mass -= fuel                                          # deplete mass
    return np.array(cum), mass


def fuel_burn(alt_ft, ts_s, mach: float, mass_init_kg: float,
              actype: str = "A320") -> tuple[float, float]:
    """(fuel kg, final mass kg) along a trajectory: the ONE fuel measure for every solver."""
    cum, mass = _burn(alt_ft, ts_s, mach, mass_init_kg, actype)
    return cum[-1], mass


def cumulative_fuel(alt_ft, ts_s, mach: float, mass_init_kg: float, actype: str) -> np.ndarray:
    """Fuel burnt up to each point (kg), same model as :func:`fuel_burn`."""
    return _burn(alt_ft, ts_s, mach, mass_init_kg, actype)[0]
