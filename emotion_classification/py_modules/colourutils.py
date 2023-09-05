from matplotlib.colors import hsv_to_rgb, to_hex


def get_n_colours(n, s=0.5, v=0.95):
    return [hsv_to_rgb((i/n, s, v)) for i in range(n)]


def extend_colour_map(data, colour_map, date_colour):
    missing_values = [x for x in data.dropna().unique() if x not in colour_map]  # All events that don't have a specified colour

    if date_colour is None:  # If the default date square colour isn't specified, we should generate this too
        new_colours = get_n_colours(len(missing_values)+1)
        date_colour = new_colours.pop()
    else:
        new_colours = get_n_colours(len(missing_values))

    new_colours_map = {x[0]: x[1] for x in zip(missing_values, new_colours)}  # Match the events and newly generated colours
    colour_map = {k: to_hex(c) for k, c in {**colour_map, **new_colours_map}.items()}  # Concat dicts and convert all colours to hex
    return colour_map, to_hex(date_colour)
