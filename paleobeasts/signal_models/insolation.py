def daily_insolation(lat, day, orb=const.orb_present, S0=const.S0, day_type=1,
                     days_per_year=const.days_per_year):
    """Compute daily average insolation given latitude, time of year and orbital parameters.

    Orbital parameters can be interpolated to any time in the last 5 Myears with
    ``climlab.solar.orbital.OrbitalTable`` (see example above).

    Longer orbital tables are available with ``climlab.solar.orbital.LongOrbitalTable``

    Inputs can be scalar, ``numpy.ndarray``, or ``xarray.DataArray``.

    The return value will be ``numpy.ndarray`` if **all** the inputs are ``numpy``.
    Otherwise ``xarray.DataArray``.

    **Function-call argument** \n

    :param array lat:       Latitude in degrees (-90 to 90).
    :param array day:       Indicator of time of year. See argument ``day_type``
                            for details about format.
    :param dict orb:        a dictionary with three members (as provided by
                            ``climlab.solar.orbital.OrbitalTable``)

                            * ``'ecc'`` - eccentricity

                                * unit: dimensionless
                                * default value: ``0.017236``

                            * ``'long_peri'`` - longitude of perihelion (precession angle)

                                * unit: degrees
                                * default value: ``281.37``

                            * ``'obliquity'`` - obliquity angle

                                * unit: degrees
                                * default value: ``23.446``

    :param float S0:        solar constant                                  \n
                            - unit: :math:`\\textrm{W}/\\textrm{m}^2`       \n
                            - default value: ``1365.2``
    :param int day_type:    Convention for specifying time of year (+/- 1,2) [optional].

                            *day_type=1* (default):
                             day input is calendar day (1-365.24), where day 1
                             is January first. The calendar is referenced to the
                             vernal equinox which always occurs at day 80.

                            *day_type=2:*
                             day input is solar longitude (0-360 degrees). Solar
                             longitude is the angle of the Earth's orbit measured from spring
                             equinox (21 March). Note that calendar days and solar longitude are
                             not linearly related because, by Kepler's Second Law, Earth's
                             angular velocity varies according to its distance from the sun.
    :raises: :exc:`ValueError`
                            if day_type is neither 1 nor 2
    :returns:               Daily average solar radiation in unit
                            :math:`\\textrm{W}/\\textrm{m}^2`.

                            Dimensions of output are ``(lat.size, day.size, ecc.size)``
    :rtype:                 array


    Code is fully vectorized to handle array input for all arguments.       \n
    Orbital arguments should all have the same sizes.
    This is automatic if computed from
    :func:`~climlab.solar.orbital.OrbitalTable.lookup_parameters`

        For more information about computation of solar insolation see the
        :ref:`Tutorial` chapter.

     """
    phi, day, ecc, long_peri, obliquity, input_is_xarray, _ignored = _standardize_inputs(lat, day, orb)
    # lambda_long (solar longitude) is the angular distance along Earth's orbit
    # measured from spring equinox (21 March)
    if day_type == 1:  # calendar days
        lambda_long = deg2rad(solar_longitude(day, orb, days_per_year))
    elif day_type == 2:  # solar longitude (1-360) is specified in input, no need to convert days to longitude
        lambda_long = deg2rad(day)
    else:
        raise ValueError('Invalid day_type.')

    # Compute declination angle of the sun
    delta = arcsin(sin(obliquity) * sin(lambda_long))
    # Compute Ho, the hour angle at sunrise / sunset
    #  Check for no sunrise or no sunset: Berger 1978 eqn (8),(9)
    Ho = xr.where(abs(delta) - pi / 2 + abs(phi) < 0.,  # there is sunset/sunrise
                  arccos(-tan(phi) * tan(delta)),
                  # otherwise figure out if it's all night or all day
                  xr.where(phi * delta > 0., pi, 0.))
    # this is not really the daily average cosine of the zenith angle...
    #  it's the integral from sunrise to sunset of that quantity...
    coszen = Ho * sin(phi) * sin(delta) + cos(phi) * cos(delta) * sin(Ho)
    # Compute daily average insolation
    Fsw = _compute_insolation_Berger(S0, ecc, lambda_long, long_peri, coszen)
    if input_is_xarray:
        return Fsw
    else:
        # Dimensional ordering consistent with previous numpy code
        return Fsw.transpose().values


def _compute_insolation_Berger(S0, ecc, lambda_long, long_peri, coszen):
    # Compute insolation: Berger 1978 eq (10)
    return S0 / pi * coszen * (1 + ecc * cos(lambda_long - long_peri)) ** 2 / (1 - ecc ** 2) ** 2


def instant_insolation(lat, day, lon=0., orb=const.orb_present, S0=const.S0,
                       days_per_year=const.days_per_year):
    """Compute instantaneous insolation given latitude, longitude, time of year and orbital parameters.

    Orbital parameters can be interpolated to any time in the last 5 Myears with
    ``climlab.solar.orbital.OrbitalTable`` (see example above).

    Longer orbital tables are available with ``climlab.solar.orbital.LongOrbitalTable``

    Inputs can be scalar, ``numpy.ndarray``, or ``xarray.DataArray``.

    The return value will be ``numpy.ndarray`` if **all** the inputs are ``numpy``.
    Otherwise ``xarray.DataArray``.

    **Function-call argument** \n

    :param array lat:       Latitude in degrees (-90 to 90).
    :param array day:       Indicator of time of year. Format is calendar day (1-365.24), where day 1
                             is January first. The calendar is referenced to the
                             vernal equinox which always occurs at day 80.
    :param array lon:       Longitude in degrees (0 to 360), optional. Defaults to zero.
    :param dict orb:        a dictionary with three members (as provided by
                            ``climlab.solar.orbital.OrbitalTable``)

                            * ``'ecc'`` - eccentricity

                                * unit: dimensionless
                                * default value: ``0.017236``

                            * ``'long_peri'`` - longitude of perihelion (precession angle)

                                * unit: degrees
                                * default value: ``281.37``

                            * ``'obliquity'`` - obliquity angle

                                * unit: degrees
                                * default value: ``23.446``

    :param float S0:        solar constant                                  \n
                            - unit: :math:`\\textrm{W}/\\textrm{m}^2`       \n
                            - default value: ``1365.2``
    :returns:               Daily average solar radiation in unit
                            :math:`\\textrm{W}/\\textrm{m}^2`.

                            Dimensions of output are ``(lat.size, day.size, ecc.size)``
    :rtype:                 array


    Code is fully vectorized to handle array input for all arguments.       \n
    Orbital arguments should all have the same sizes.
    This is automatic if computed from
    :func:`~climlab.solar.orbital.OrbitalTable.lookup_parameters`

        For more information about computation of solar insolation see the
        :ref:`Tutorial` chapter.

     """
    phi, day, ecc, long_peri, obliquity, input_is_xarray, lam = _standardize_inputs(lat, day, orb, lon)
    # lambda_long (solar longitude) is the angular distance along Earth's orbit
    # measured from spring equinox (21 March)
    lambda_long = deg2rad(solar_longitude(day, orb, days_per_year))
    # Compute declination angle of the sun
    delta = arcsin(sin(obliquity) * sin(lambda_long))

    # np.fmod(day, 1.0) finds the "fractional" time of day with a range of [0,1)
    # where 0 is midnight, and 0.9999... is 23:59. lon/360 converts longitude
    # to time since moving along the longitude axis produces the same effect as
    # changing time while keeping longitude the same. the fractional time and
    # fractional longitude are added together since they now both represent
    # longitude/time of day. This lets us calculate the local solar time (in
    # "fractional" units) and then convert to hour angle. The -0.5 is included
    # in order to assert that noon occurs when the sun is overhead (so h=0 at
    # LST=0.5 aka time=noon).
    LST = np.fmod((np.fmod(day, 1.0) + (lam / (2 * pi))), 1.0)
    # hour angle in rad
    h = (LST - 0.5) * 2 * pi
    # instantaneous cosine of solar zenith angle
    coszen = (sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(h)) * pi
    # Compute insolation
    Fsw = _compute_insolation_Berger(S0, ecc, lambda_long, long_peri, coszen)
    # assert |h|<=Ho, i.e. it is day time (same as checking Fsw >= 0)
    Fsw = np.maximum(Fsw, 0.0)
    if input_is_xarray:
        return Fsw
    else:
        # Dimensional ordering consistent with previous numpy code
        return Fsw.transpose().values