import random

import networkx as nx
import numpy as np
import pandas as pd
from bokeh import palettes
from bokeh.models import Circle, GraphRenderer, MultiLine, Rect, StaticLayoutProvider
from bokeh.plotting import figure


def _get_trips(trips_df, start_time, duration):
    """Return trips that start and finish inside a time window."""
    end_time = start_time + duration
    trips = trips_df.copy()
    return trips[
        (trips.start_time >= start_time)
        & (trips.start_time + trips.trip_time <= end_time)
    ].copy()


def _compute_time(available_taxi, pickup_request, nodes_df):
    """Estimate travel time between an available taxi and a pickup request."""
    lat1 = nodes_df.loc[available_taxi[0], "lat"]
    lon1 = nodes_df.loc[available_taxi[0], "lon"]
    lat2 = nodes_df.loc[pickup_request[0], "lat"]
    lon2 = nodes_df.loc[pickup_request[0], "lon"]
    radius_earth_km = 6371
    distance = 0.01 + radius_earth_km * (abs(lon1 - lon2) + abs(lat1 - lat2)) * np.pi / 180
    return 2 * distance


def prepare_taxi_assignment_nodes(
    available_taxi_window_start=1123,
    available_taxi_window_duration=5,
    gap_interval_duration=0,
    pickup_window_duration=5,
    slide_available_taxi_window=False,
    trips_path="data/2013-09-01_trip_data_manhattan.csv",
    nodes_path="data/nyc_nodes_manhattan.csv",
):
    """Load taxi data and create available taxi and pickup request nodes."""
    trips_df = pd.read_csv(trips_path).drop(columns="id")
    nodes_df = pd.read_csv(nodes_path).drop(columns="Unnamed: 0")

    pickup_start = available_taxi_window_start + available_taxi_window_duration + gap_interval_duration
    pickup_request_trips = _get_trips(trips_df, pickup_start, pickup_window_duration)
    pickup_requests = [
        (int(row["start_node"]), row["start_time"], index, "PU")
        for index, row in pickup_request_trips.iterrows()
    ]

    sliding = True
    while sliding:
        available_taxi_trips = _get_trips(trips_df, available_taxi_window_start, available_taxi_window_duration)
        available_taxis = [
            (int(row["end_node"]), row["start_time"] + row["trip_time"], index, "DO")
            for index, row in available_taxi_trips.iterrows()
        ]

        if (
            len(available_taxis) >= len(pickup_requests)
            or not slide_available_taxi_window
            or available_taxi_window_start <= 0
        ):
            sliding = False
        else:
            available_taxi_window_start -= 1
            available_taxi_window_duration += 1

    available_taxis.sort(key=lambda node: node[1])
    pickup_requests.sort(key=lambda node: node[1])

    identifier = 0
    for i, node in enumerate(available_taxis):
        available_taxis[i] = tuple(list(node) + [identifier])
        identifier += 1
    for i, node in enumerate(pickup_requests):
        pickup_requests[i] = tuple(list(node) + [identifier])
        identifier += 1

    return trips_df, nodes_df, available_taxis, pickup_requests, identifier


def build_taxi_assignment_arcs(
    available_taxis,
    pickup_requests,
    nodes_df,
    travel_window,
    pickup_window_duration=float("inf"),
    use_travel_time_as_cost=True,
):
    """Create feasible taxi-to-pickup arcs for a given travel-time window."""
    arcs = []
    for available_taxi in available_taxis:
        for pickup_request in pickup_requests:
            if pickup_request[1] >= available_taxi[1]:
                time = _compute_time(available_taxi, pickup_request, nodes_df)
                arrival_time = available_taxi[1] + time
                latest_valid_arrival = pickup_request[1] + pickup_window_duration
                if arrival_time <= latest_valid_arrival and time <= travel_window:
                    cost = time if use_travel_time_as_cost else arrival_time - pickup_request[1]
                    cost = 0 if cost <= 0 else int(cost * 1000)
                    arcs.append((available_taxi, pickup_request, cost))

    return arcs


def prepare_taxi_assignment_data(
    travel_window,
    available_taxi_window_start=1123,
    available_taxi_window_duration=5,
    gap_interval_duration=0,
    pickup_window_duration=5,
    use_travel_time_as_cost=True,
    slide_available_taxi_window=False,
    trips_path="data/2013-09-01_trip_data_manhattan.csv",
    nodes_path="data/nyc_nodes_manhattan.csv",
):
    """Load taxi data and create available taxis, pickup requests, and feasible arcs."""
    trips_df, nodes_df, available_taxis, pickup_requests, identifier = prepare_taxi_assignment_nodes(
        available_taxi_window_start=available_taxi_window_start,
        available_taxi_window_duration=available_taxi_window_duration,
        gap_interval_duration=gap_interval_duration,
        pickup_window_duration=pickup_window_duration,
        slide_available_taxi_window=slide_available_taxi_window,
        trips_path=trips_path,
        nodes_path=nodes_path,
    )
    arcs = build_taxi_assignment_arcs(
        available_taxis,
        pickup_requests,
        nodes_df,
        travel_window,
        use_travel_time_as_cost=use_travel_time_as_cost,
    )
    return trips_df, nodes_df, available_taxis, pickup_requests, arcs, identifier


def build_all_taxi_to_pickup_arcs(available_taxis, pickup_requests, nodes_df):
    """Create one travel-time arc for every available taxi and pickup request pair."""
    arcs = []
    for available_taxi in available_taxis:
        for pickup_request in pickup_requests:
            time = _compute_time(available_taxi, pickup_request, nodes_df)
            cost = int(max(time, 0) * 1000)
            arcs.append((available_taxi, pickup_request, cost))
    return arcs


def random_taxi_matching(arcs, seed=None):
    """Return a random greedy feasible matching from taxi-to-pickup arcs."""
    rng = random.Random(seed)
    shuffled_arcs = arcs.copy()
    rng.shuffle(shuffled_arcs)

    matching = []
    used_taxis = set()
    used_requests = set()

    for taxi_node, request_node, cost in shuffled_arcs:
        if taxi_node in used_taxis or request_node in used_requests:
            continue
        matching.append((taxi_node, request_node, cost))
        used_taxis.add(taxi_node)
        used_requests.add(request_node)

    return matching


def random_taxi_matching_summary(arcs, trials=100):
    """Run random greedy matching several times and summarize the matching sizes."""
    sizes = [len(random_taxi_matching(arcs, seed=trial)) for trial in range(trials)]
    return {
        "average": sum(sizes) / len(sizes),
        "minimum": min(sizes),
        "maximum": max(sizes),
        "sizes": sizes,
    }


def greedy_taxi_matching(arcs):
    """Return a greedy matching that repeatedly picks the shortest available arc."""
    matching = []
    used_taxis = set()
    used_requests = set()

    for taxi, pickup_request, cost in sorted(arcs, key=lambda edge: edge[2]):
        if taxi in used_taxis or pickup_request in used_requests:
            continue
        matching.append((taxi, pickup_request, cost))
        used_taxis.add(taxi)
        used_requests.add(pickup_request)

    return matching


def optimal_taxi_matching(available_taxis, pickup_requests, arcs):
    """Return a maximum-cardinality matching from the feasible taxi-to-pickup arcs."""
    B = nx.Graph()
    B.add_nodes_from(available_taxis, bipartite=0)
    B.add_nodes_from(pickup_requests, bipartite=1)
    B.add_edges_from([arc[:2] for arc in arcs])

    match = nx.bipartite.maximum_matching(B, available_taxis)
    return [
        (taxi, pickup_request)
        for taxi, pickup_request in match.items()
        if taxi in available_taxis
    ]


def greedy_passengers_under_threshold(available_taxis, pickup_requests, nodes_df, pickup_time_threshold):
    """Print the number of passengers served by the greedy matching under a threshold."""
    arcs = build_taxi_assignment_arcs(
        available_taxis,
        pickup_requests,
        nodes_df,
        pickup_time_threshold,
    )
    matching = greedy_taxi_matching(arcs)
    print(f"Greedy passengers under {pickup_time_threshold} minutes: {len(matching)}")
    return


def optimal_passengers_under_threshold(available_taxis, pickup_requests, nodes_df, pickup_time_threshold):
    """Print the number of passengers served by an optimal matching under a threshold."""
    arcs = build_taxi_assignment_arcs(
        available_taxis,
        pickup_requests,
        nodes_df,
        pickup_time_threshold,
    )
    matching = optimal_taxi_matching(available_taxis, pickup_requests, arcs)
    print(f"Optimal passengers under {pickup_time_threshold} minutes: {len(matching)}")
    return


def _map_bounds(nodes):
    min_x, max_x = -8240298.040280505, -8230749.832964136
    min_y, max_y = 4968176.938664163, 4984234.650659162

    if len(nodes) >= 1:
        min_x, max_x = min(nodes.x) - 1000, max(nodes.x) + 1000
        min_y, max_y = min(nodes.y) - 1000, max(nodes.y) + 1000

    return min_x, max_x, min_y, max_y


def _base_taxi_map(nodes_df, node_ids, title):
    nodes = nodes_df.loc[list(set(node_ids))]
    min_x, max_x, min_y, max_y = _map_bounds(nodes)

    plot = figure(
        x_range=(min_x, max_x),
        y_range=(min_y, max_y),
        x_axis_type="mercator",
        y_axis_type="mercator",
        title=title,
        width=600,
        height=470,
        tools="",
        toolbar_location=None,
    )
    plot.add_tile("CartoDB Positron retina")

    graph_layout = dict(zip(nodes.name.values.tolist(), zip(nodes.x.values.tolist(), nodes.y.values.tolist())))
    return plot, graph_layout


def plot_taxi_locations(starts, ends, nodes_df, title):
    """Plot available taxi and pickup request locations on a Manhattan map."""
    start_nodes = [node[0] for node in starts]
    end_nodes = [node[0] for node in ends]

    plot, graph_layout = _base_taxi_map(nodes_df, start_nodes + end_nodes, title)

    graph = GraphRenderer()
    graph.node_renderer.data_source.add(start_nodes, "index")
    graph.node_renderer.data_source.add(["green"] * len(start_nodes), "start_colors")
    graph.node_renderer.glyph = Circle(size=7, line_width=0, fill_alpha=1, fill_color="start_colors")
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    end_graph = GraphRenderer()
    end_graph.node_renderer.data_source.add(end_nodes, "index")
    end_graph.node_renderer.data_source.add(["red"] * len(end_nodes), "end_colors")
    end_graph.node_renderer.glyph = Rect(
        height=7,
        width=7,
        height_units="screen",
        width_units="screen",
        line_width=0,
        fill_alpha=1,
        fill_color="end_colors",
    )
    end_graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    plot.renderers.append(graph)
    plot.renderers.append(end_graph)
    return plot


def plot_taxi_route(paths, nodes_df, title="Taxi Routes"):
    """Plot taxi assignment routes on a Manhattan map."""
    start = []
    end = []
    color = []
    alpha = []
    start_nodes = []
    start_colors = []
    end_nodes = []
    end_colors = []

    colors = palettes.Category10[10]
    c = 0

    for path in paths:
        start_nodes.append(path[0][0][0])
        start_colors.append(colors[c])
        end_nodes.append(path[0][1][0])
        end_colors.append(colors[c])

        for comp in path:
            start.append(comp[0][0])
            end.append(comp[1][0])
            alpha.append({True: 0.3, False: 1}[comp[2]])
            color.append(colors[c])

        c = c + 1 if c < len(colors) - 1 else 0

    plot, graph_layout = _base_taxi_map(nodes_df, start + end, title)

    graph = GraphRenderer()
    graph.node_renderer.data_source.add(start_nodes, "index")
    graph.node_renderer.data_source.add(start_colors, "start_colors")
    graph.node_renderer.glyph = Circle(size=7, line_width=0, fill_alpha=1, fill_color="start_colors")
    graph.edge_renderer.data_source.data = {
        "start": list(start),
        "end": list(end),
        "color": list(color),
        "alpha": list(alpha),
    }
    graph.edge_renderer.glyph = MultiLine(
        line_color="color",
        line_alpha="alpha",
        line_width=3,
        line_cap="round",
    )
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    end_graph = GraphRenderer()
    end_graph.node_renderer.data_source.add(end_nodes, "index")
    end_graph.node_renderer.data_source.add(end_colors, "end_colors")
    end_graph.node_renderer.glyph = Rect(
        height=7,
        width=7,
        height_units="screen",
        width_units="screen",
        line_width=0,
        fill_alpha=1,
        fill_color="end_colors",
    )
    end_graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    plot.renderers.append(graph)
    plot.renderers.append(end_graph)
    return plot
