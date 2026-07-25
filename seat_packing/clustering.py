# same imports as in Jupyter Notebook
import pandas as pd
import numpy as np
import math, itertools
import matplotlib.pyplot as plt
import networkx as nx
from ortools.linear_solver import pywraplp as OR
import shapely, shapely.affinity
from shapely.geometry import Polygon, Point
from IPython.display import display

weekday_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

def two_hour_day_vector(taxi_days, day_id):
    counts_15_min = np.array(taxi_days.at[day_id, 'count_vector'], dtype=float)
    two_hour_counts = counts_15_min.reshape(12, 8).sum(axis=1)
    return two_hour_counts

def time_block_label(block):
    if block == 0:
        return '12 AM'
    if block < 6:
        return '%d AM' % (2 * block)
    if block == 6:
        return '12 PM'
    return '%d PM' % (2 * (block - 6))

def extract_kmedoid_clusters(x, y, days):
    centers = [i for i in days if y[i].solution_value() > 0.5]
    result = {i: [] for i in centers}
    for j in days:
        for i in centers:
            if x[i, j].solution_value() > 0.5:
                result[i].append(j)
    return result

def kmedoid_metrics(clusters_dict, cost, features, taxi_days):
    total = 0
    worst = 0
    rows = []

    for center_day, members in clusters_dict.items():
        distances = [cost[center_day, j] for j in members]
        total += sum(distances)
        worst = max(worst, max(distances))
        weekday_counts = taxi_days.loc[members, 'weekday'].map(weekday_names).value_counts().to_dict()
        peak = int(np.argmax(features[center_day]))
        rows.append({
            'center_day': center_day,
            'center_date': taxi_days.at[center_day, 'date'],
            'size': len(members),
            'avg_distance': round(float(np.mean(distances)), 4),
            'max_distance': round(float(max(distances)), 4),
            'peak_hour': time_block_label(peak),
            'weekday_counts': weekday_counts,
            'sample_dates': ', '.join(taxi_days.loc[members[:6], 'date'].tolist())
        })

    return pd.DataFrame(rows), total, worst

def plot_kmedoid_centers(clusters_dict, features, time_blocks, title, taxi_days):
    plt.figure(figsize=(10, 4))
    for center_day in clusters_dict:
        plt.plot(time_blocks, features[center_day], label='Center %s (%s)' % (center_day, taxi_days.at[center_day, 'date']))
    plt.xticks(range(0, 12, 2), [time_block_label(h) for h in range(0, 12, 2)])
    plt.xlabel('Time block')
    plt.ylabel('Two-hour pickup count')
    plt.title(title)
    plt.legend()
    plt.show()

def pca_2d(feature_matrix):
    """Project rows of feature_matrix onto their first two principal components.

    Returns the projected coordinates and the fraction of variance each
    component explains.
    """
    centered = feature_matrix - feature_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ Vt[:2].T
    explained = S**2 / (S**2).sum()
    return coords, explained[:2]


def plot_kmedoid_pca(clusters_dict, features, title, dates=None):
    """Scatter every day in 2-D PCA space, colored by the cluster it belongs to.

    Pass dates as a {day_id: date_string} dictionary to label the legend by date.
    """
    days = sorted(features)
    day_index = {day: row for row, day in enumerate(days)}
    coords, explained = pca_2d(np.array([features[day] for day in days]))

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    for c, (center_day, members) in enumerate(clusters_dict.items()):
        rows = [day_index[j] for j in members]
        name = 'day %s' % center_day if dates is None else '%s (day %s)' % (dates[center_day], center_day)
        plt.scatter(coords[rows, 0], coords[rows, 1], s=30, alpha=0.7, color=colors[c % 10],
                    label='Cluster centered on %s, %d day%s' % (name, len(members), '' if len(members) == 1 else 's'))
        center_row = day_index[center_day]
        plt.scatter(coords[center_row, 0], coords[center_row, 1], s=350, marker='*',
                    color=colors[c % 10], edgecolor='black', linewidth=1, zorder=3)

    plt.xlabel('First principal component (%.0f%% of variance)' % (100 * explained[0]))
    plt.ylabel('Second principal component (%.0f%% of variance)' % (100 * explained[1]))
    plt.title(title)
    plt.legend()
    plt.show()


def solve_and_analyze_kmedoid(label, m, x, y, days, time_blocks, cost, features, taxi_days):
    status = m.Solve()
    if status == OR.Solver.OPTIMAL:
        print('Solver status: optimal solution found')
    elif status == OR.Solver.FEASIBLE:
        print('Solver status: feasible solution found before the time limit')
        print('This solution is useful to analyze, but it has not been proved optimal.')
    else:
        print('Solver status:', status)
        print('The model may be missing a constraint or objective.')
        return None

    clusters_dict = extract_kmedoid_clusters(x, y, days)
    summary, total, worst = kmedoid_metrics(clusters_dict, cost, features, taxi_days)

    print(label)
    print('Solver objective value:', round(m.Objective().Value(), 4))
    print('Total assignment cost:', round(total, 4))
    print('Worst single-day distance:', round(worst, 4))
    pd.set_option("display.max_colwidth", None)
    display(summary)
    plot_kmedoid_centers(clusters_dict, features, time_blocks, label, taxi_days)

    return {
        'label': label,
        'clusters': clusters_dict,
        'summary': summary,
        'total_distance': total,
        'max_radius': worst,
        'sizes': sorted(summary['size'].tolist())
    }
