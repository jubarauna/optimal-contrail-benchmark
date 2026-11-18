from openap import contrail, aero


def agg_conditions(flight):
    f = flight.assign(
        rhi=lambda d: contrail.relative_humidity(d.specific_humidity,
                                                 aero.pressure(d.altitude * aero.ft),
                                                 d.temperature, to="ice"),
        crit_temp=lambda d: contrail.critical_temperature_water(aero.pressure(d.altitude * aero.ft)),
        sac=lambda d: d.temperature < d.crit_temp,
        issr=lambda d: d.rhi > 1,
        persistent=lambda d: d.sac & d.issr
    )
    return f
