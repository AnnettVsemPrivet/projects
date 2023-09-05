import calendar

import pandas as pd

from colourutils import extend_colour_map


def extend_data_range(data):
    """
    Extends the index of the given Series so that it has daily values, starting from the 1st of the earliest month and
    ending on the last day of the latest month.
    :param data: The Series to be extended with a datetime index
    :return: The Series with an extended daily index
    """
    earliest_date = data.index.min()
    first_month_start = pd.Timestamp(year=earliest_date.year, month=earliest_date.month, day=1)

    latest_date = data.index.max()
    _, last_date_of_month = calendar.monthrange(latest_date.year, latest_date.month)
    last_month_end = pd.Timestamp(year=latest_date.year, month=latest_date.month, day=last_date_of_month)

    if first_month_start not in data:
        data[first_month_start] = None
    if last_month_end not in data:
        data[last_month_end] = None

    data = data.groupby(data.index).agg(list)  # Collate multiple values for a given date as a list
    data = data.sort_index().asfreq('D')  # Resample as daily data
    data = data.explode()  # Unpack the list over multiple rows

    return data


def apply_colour_map(data, colour_map, date_colour, exclude_colour, min_date, max_date):
    """
    Converts the events in the Series to a colour value using the colour map for mapped events, and the default date
    and exclude colours for other dates.
    :param data: The events Series
    :param colour_map: A map from event to colour
    :param date_colour: The default colour for a date square
    :param exclude_colour: The colour for a date square which falls outside of the start/end date
    :param min_date: Dates with null data before this date will use the exclude colour
    :param max_date: Dates with null data after this date will use the exclude colour
    :return: A Series with a colour value for every date
    """
    data = data.map(colour_map)
    data[pd.isna(data) & ((data.index < min_date) | (data.index > max_date))] = exclude_colour
    data = data.fillna(date_colour)
    return data


def colour_data(data,
                colour_map,
                date_colour,
                exclude_colour,
                min_date,
                max_date,
                generate_missing_colours,
                strict_exclude):
    """
    Convert an events Series to daily colour values, using the other parameters to control colours and mapping behaviour
    :param data: Series with datetime index and event data
    :param colour_map: Map from event to colour, can be empty
    :param date_colour: The colour for a date square with no events
    :param exclude_colour: The colour for a date square outside of the start/end dates
    :param min_date: Date squares before this date use the exclude_colour
    :param max_date: Date squares after this date use the exclude_colour
    :param generate_missing_colours: Whether or not to generate colours for events not present in the given colour_map
    :param strict_exclude: Should we apply colours for events that fall outside of the start/end dates
    :return: A Series with a daily datetime index and colour values
    """

    if strict_exclude:  # If we should ignore events outside the provided date ranges
        data[data.index < pd.to_datetime(min_date, format='%Y-%m-%d')] = None
        data[data.index > pd.to_datetime(max_date, format='%Y-%m-%d')] = None

    data = extend_data_range(data)

    if min_date is None:
        min_date = data.index.min()
    if max_date is None:
        max_date = data.index.max()

    if generate_missing_colours:
        colour_map, date_colour = extend_colour_map(data, colour_map, date_colour)
    elif date_colour is None:
        # Hex conversion?
        date_colour = '#ed97ca'  # default value, light pink

    coloured_data = apply_colour_map(data, colour_map, date_colour, exclude_colour, min_date, max_date)

    return coloured_data, colour_map

