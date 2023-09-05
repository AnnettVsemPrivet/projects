import calendar
import math
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from datautils import colour_data
from dateutils import count_months, get_week_of_month


def _get_date_square_coordinates(day_of_week, week_of_month, scale, i=1, n=1):
    width = scale / 2

    x_base = scale * day_of_week
    x_start = x_base + (((i - 1) / n) * width)
    x_end = x_base + ((i / n) * width)
    y_start = scale * week_of_month
    y_end = (scale * week_of_month) + (scale / 2)

    bottom_left = (x_start, y_start)
    top_left = (x_start, y_end)
    top_right = (x_end, y_end)
    bottom_right = (x_end, y_start)

    centre = (((x_start + x_end) / 2), ((y_start + y_end) / 2))

    return [bottom_left, top_left, top_right, bottom_right], centre


def _draw_date_square(square_date, data, ax, text_colour, scale):
    day_of_week = square_date.weekday()
    week_of_month = get_week_of_month(square_date)

    _, date_centre = _get_date_square_coordinates(day_of_week, week_of_month, scale)
    
    values = data[[square_date]].values
    n = len(values)

    for i, val in enumerate(values):
        corners, _ = _get_date_square_coordinates(day_of_week, week_of_month, scale, i+1, n)
        shape = plt.Polygon(corners, color=val)
        ax.add_patch(shape)
        
    ax.annotate(str(square_date.day), date_centre, color=text_colour, weight='bold', fontsize=scale * 0.75, ha='center', va='center')


def _setup_weekday_axis(ax, scale):
    secax = ax.secondary_xaxis('top', functions=(
        lambda x: (x - (scale/4)) / scale,  # num to day
        lambda x: (x * scale) + (scale / 4)  # day to num
    ))
    secax.set_xticks(range(7))
    secax.set_xticklabels(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'], fontsize=scale)


def draw_month_calendar(data, year, month, ax, text_colour, scale):
    """
    Draw a calendar for a given year-month on a given axis, using a provided Series of dates and colours.

    :param data: A Series which must contain a colour value for every date in the month.
    :param year: The year (YYYY).
    :param month: The month (MM).
    :param ax: The axis to plot on.
    :param text_colour: The text colour for the date numbers.
    :return: None
    """
    month_start_day, month_num_days = calendar.monthrange(year, month)
    for day in range(1, month_num_days+1):
        current_date = datetime(year, month, day)
        _draw_date_square(current_date, data, ax, text_colour, scale)

    _setup_weekday_axis(ax, scale)
    ax.axis('scaled')
    ax.set_title(calendar.month_name[month] + ' ' + str(year), pad=20, fontsize=scale)
    ax.axis('off')
    ax.plot()


def _draw_legend(fig, colour_map, scale):
    markers = [plt.Line2D([0,0], [0,0], color=c, marker='o', linestyle='') for c in colour_map.values()]
    fig.legend(markers, colour_map.keys(), numpoints=1, markerscale=scale/3, fontsize=scale*2, loc='lower center',
               bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=fig.transFigure, ncol=int(math.sqrt(len(colour_map))))


def draw_colour_calendar(data,
                         colour_map=None,
                         generate_colours=True,
                         months_per_row=3,
                         date_colour=None,
                         text_colour='w',
                         exclude_colour='tab:gray',
                         strict_exclude=False,
                         min_date=None,
                         max_date=None,
                         legend=True):
    """
    Draw monthly calendar(s) with dates from the provided data highlighted as discrete events. Optional parameters
    control the appearance of the resultant figure.

    :param data: A Series of events with a datetime index.
    :param colour_map: A dictionary mapping events to colours.
    :param generate_colours: Whether to generate colours for events which don't have a mapping provided in
        colour_map.
    :param months_per_row: When displaying more than one month, how many to fit per row.
    :param date_colour: The default colour for a date square with no events. If not specified, a colour will be
        generated in the same spectrum as those for missing events.
    :param text_colour: The colour for the date numbers on the date squares.
    :param exclude_colour: The colour to shade dates which fall outside of min_date or max_date.
    :param strict_exclude: Whether to exclude events which fall outside of min_date or max_date.
    :param min_date: Dates before the min_date are coloured with the exclude_colour.
    :param max_date: Dates after the max_date are coloured with the exclude_colour.
    :param legend: Whether to display the event-to-colour legend.
    :return: None
    """
    scale = 20  # Arbitrary value

    # Setup parameters
    data.index = pd.to_datetime(data.index)
    data.index = data.index.normalize()  # Remove time portion of datetime index

    if colour_map is None:
        colour_map = {}

    # Convert data to daily colour values
    data, colour_map = colour_data(data, colour_map, date_colour, exclude_colour, min_date, max_date, generate_colours, strict_exclude)

    first_date = data.index.min()
    last_date = data.index.max()

    num_months = count_months(first_date, last_date)

    fig, axs = plt.subplots(math.ceil(num_months/months_per_row), min(num_months, months_per_row), sharex=True, sharey=True, squeeze=False)

    total_axs = len(axs) * len(axs[0])
    unused_axs = total_axs - num_months
    for u in range(unused_axs):
        axs[len(axs)-1][(months_per_row-1)-u].remove()

    month_counter = 0
    for year in range(first_date.year, last_date.year + 1):
        start_month = 1
        end_month = 12
        if year == first_date.year:
            start_month = first_date.month
        if year == last_date.year:
            end_month = last_date.month

        for month in range(start_month, end_month+1):
            axs_x = int(month_counter / months_per_row)
            axs_y = month_counter % months_per_row
            draw_month_calendar(data, year, month, axs[axs_x][axs_y], text_colour, scale)
            month_counter = month_counter + 1

    axs[0][0].invert_yaxis()

    fig.set_size_inches(10*min(num_months, months_per_row), 10*len(axs))
    if legend:
        _draw_legend(fig, colour_map, scale)
    fig.set_dpi(200)
    fig.tight_layout()
    plt.savefig('../bot_data/photo.png', bbox_inches='tight')
