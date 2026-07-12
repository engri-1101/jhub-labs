import networkx as nx
from bokeh.models import HoverTool
from vinal.plot import _graph_plot


def get_simple_bipartite_graph():
    """Return the toy bipartite graph used in Part 1 of the lab."""
    left_nodes = ["L1", "L2", "L3", "L4"]
    right_nodes = ["R1", "R2", "R3", "R4"]
    compatibility_edges = [
        ("L1", "R2"),
        ("L2", "R2"),
        ("L2", "R3"),
        ("L3", "R1"),
        ("L3", "R2"),
        ("L3", "R3"),
        ("L4", "R3"),
        ("L4", "R4"),
    ]

    B = nx.Graph()
    B.add_nodes_from(left_nodes, bipartite=0)
    B.add_nodes_from(right_nodes, bipartite=1)
    B.add_edges_from(compatibility_edges)

    return B, left_nodes, right_nodes, compatibility_edges


def plot_bipartite(B, left_nodes, right_nodes, matching_edges=None):
    """Return a small bipartite graph plot using the default vinal graph style."""
    if matching_edges is None:
        matching_edges = []

    index_to_node = {i: node for i, node in enumerate(left_nodes)}
    index_to_node.update({len(left_nodes) + i: node for i, node in enumerate(right_nodes)})
    matching_edges = [
        (index_to_node[u], index_to_node[v]) if isinstance(u, int) and isinstance(v, int) else (u, v)
        for u, v in matching_edges
    ]
    matching_edges = set(matching_edges)
    matching_edges = matching_edges | {(v, u) for u, v in matching_edges}

    G = nx.Graph()
    for i, node in enumerate(left_nodes):
        G.add_node(node, x=0, y=len(left_nodes) - i, name="")
    for i, node in enumerate(right_nodes):
        G.add_node(node, x=4, y=len(right_nodes) - i, name="")
    G.add_edges_from(B.edges())

    highlighted_edges = [
        edge for edge in G.edges() if edge in matching_edges or (edge[1], edge[0]) in matching_edges
    ]
    height = max(350, 55 * max(len(left_nodes), len(right_nodes)))
    plot = _graph_plot(
        G,
        edges=highlighted_edges,
        show_all_edges=True,
        show_labels=False,
        width=600,
        height=height,
    )
    figure = plot.children[0][0]
    for tool in figure.tools:
        if isinstance(tool, HoverTool):
            tool.tooltips = [("Index", "$index")]
    return plot


def check_student_matching(B, left_nodes, right_nodes, student_matching):
    """Highlight a proposed indexed matching and print whether it is valid."""
    index_to_node = {i: node for i, node in enumerate(left_nodes)}
    index_to_node.update({len(left_nodes) + i: node for i, node in enumerate(right_nodes)})

    highlighted_edges = []
    used_left = set()
    used_right = set()
    issues = []

    for pair in student_matching:
        if not isinstance(pair, tuple) or len(pair) != 2:
            issues.append(f"{pair} is not a pair like (0, 5).")
            continue

        left_idx, right_idx = pair
        if right_idx is None:
            issues.append(f"{pair} is incomplete. Replace None with a right-side index.")
            continue
        if left_idx not in range(len(left_nodes)):
            issues.append(f"{pair}: left endpoint must be one of {list(range(len(left_nodes)))}.")
            continue
        if right_idx not in range(len(left_nodes), len(left_nodes) + len(right_nodes)):
            issues.append(
                f"{pair}: right endpoint must be one of "
                f"{list(range(len(left_nodes), len(left_nodes) + len(right_nodes)))}."
            )
            continue
        if left_idx in used_left:
            issues.append(f"{pair}: left node {left_idx} is used more than once.")
        if right_idx in used_right:
            issues.append(f"{pair}: right node {right_idx} is used more than once.")

        left = index_to_node[left_idx]
        right = index_to_node[right_idx]
        edge = (left, right)
        if edge not in B.edges:
            issues.append(f"{pair}: this edge is not in the graph.")
        else:
            highlighted_edges.append(edge)
            used_left.add(left_idx)
            used_right.add(right_idx)

    from bokeh.io import show

    show(plot_bipartite(B, left_nodes, right_nodes, highlighted_edges))

    if issues:
        print("This is not a valid matching yet:")
        for issue in issues:
            print("-", issue)
        return

    print(f"This is a valid matching of size {len(highlighted_edges)}.")
    if len(highlighted_edges) == len(left_nodes):
        print("It matches every left-side node, so it is maximum.")
    return
