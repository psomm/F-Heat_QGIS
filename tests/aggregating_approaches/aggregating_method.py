# Retry with safer plotting: cap buffer distances and set axis limits to data extent to avoid overly large figure sizes.
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, Point, mapping, MultiPolygon
from shapely.ops import unary_union

# --- Parameters ---
MIN_UNITS = 5           # minimum units per group as in the paper ("five plus" rule)
PLOT_REWEIGHT = 0.05    # multiply distances by this when buildings are in the same plot
DISTANCE_FACTOR_M = 0.8 # distance factor 'm' for max-distance rule (adjustable)
BUFFER_TRIAL_MULTIPLIER = 1.0  # used when creating aggregate polygons (see code)
MAX_BUFFER_CAP = 40.0   # **safety cap** for buffer distance to avoid generating gigantic geometries in demo

random.seed(1)
np.random.seed(1)

# --- 1) Create dummy buildings ------------------
buildings = {}
b_idx = 0

def make_building(x, y, w, h, floors, use='residential', plot_id=None):
    global b_idx
    poly = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
    b = {
        'id': f'B{b_idx}',
        'geom': poly,
        'footprint_area': poly.area,
        'floors': floors,
        'use': use,
        'plot_id': plot_id if plot_id is not None else f'P{int(x//50)}',
    }
    buildings[b['id']] = b
    b_idx += 1
    return b['id']

# Cluster A (left)
make_building(0, 0, 8, 6, floors=3, use='residential', plot_id='plot_A1')   # small house
make_building(12, 0, 10, 8, floors=6, use='residential', plot_id='plot_A2') # taller res.
make_building(0, 10, 12, 10, floors=2, use='residential', plot_id='plot_A1')# same plot as first
make_building(16, 12, 8, 8, floors=1, use='non-residential', plot_id='plot_A3')
make_building(28, 4, 10, 12, floors=7, use='residential', plot_id='plot_A4')

# Cluster B (right)
make_building(60, 0, 9, 7, floors=4, use='residential', plot_id='plot_B1')
make_building(72, 2, 10, 8, floors=8, use='residential', plot_id='plot_B2')
make_building(60, 12, 14, 10, floors=5, use='non-residential', plot_id='plot_B1')  # same plot as first in B
make_building(80, 12, 6, 6, floors=1, use='residential', plot_id='plot_B3')
make_building(92, 0, 10, 20, floors=9, use='residential', plot_id='plot_B4')   # a big apartment-like

# Cluster C (isolated)
make_building(140, 0, 10, 10, floors=4, use='residential', plot_id='plot_C1')
make_building(152, 2, 12, 8, floors=6, use='residential', plot_id='plot_C2')
make_building(140, 12, 14, 10, floors=3, use='non-residential', plot_id='plot_C1') # same plot as first in C
make_building(160, 12, 8, 8, floors=2, use='residential', plot_id='plot_C3')
# make_building(172, 0, 10, 20, floors=5, use='residential', plot_id='plot_C4')   # a big apartment-like

# Cluster D (upper left, small cluster)
make_building(0, 30, 10, 10, floors=2, use='residential', plot_id='plot_D1')
make_building(12, 32, 8, 6, floors=3, use='residential', plot_id='plot_D2')
make_building(0, 42, 12, 8, floors=1, use='non-residential', plot_id='plot_D1') # same plot as first in D
make_building(160, 44, 8, 8, floors=9, use='residential', plot_id='plot_D3')

# # some isolated outlier (to test anonymization)
make_building(140, 80, 6, 6, floors=4, use='non-residential', plot_id='plot_X1')

# --- 2) Estimate units per building using the paper rules ---
def estimate_units(building):
    use = building['use']
    floors = building['floors']
    if use in ('residential', 'mixed-residential'):
        if floors <= 3:
            return 1
        elif floors <= 5:
            return 3
        else:
            return floors
    else:
        # non-residential treated as 1 unit
        return 1

for bid, b in buildings.items():
    b['est_units'] = estimate_units(b)
    # assume gross floor area = footprint_area * floors
    b['gross_floor_area'] = b['footprint_area'] * b['floors']
    # assign a dummy specific heat demand [kWh/m2*a] per building (varies randomly)
    b['specific_kwh_m2a'] = random.uniform(60, 220)  # wide range for demo
    b['total_kwh_a'] = b['specific_kwh_m2a'] * b['gross_floor_area']

# --- 3) Build complete graph with reweighted distances ---
ids = list(buildings.keys())

# compute centroid coordinates and pairwise distances
centroids = {bid: buildings[bid]['geom'].centroid for bid in ids}
def euclidean(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# complete graph
G = nx.Graph()
for bid in ids:
    G.add_node(bid, **buildings[bid])

for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        a = ids[i]; b = ids[j]
        c1 = centroids[a]; c2 = centroids[b]
        dist = euclidean(c1, c2)
        # reweight if same plot
        weight = dist * (PLOT_REWEIGHT if buildings[a]['plot_id']==buildings[b]['plot_id'] else 1.0)
        G.add_edge(a, b, weight=weight, orig_dist=dist)

# --- 4) Minimum Spanning Tree and iterative splitting ---
mst = nx.minimum_spanning_tree(G, weight='weight')

# We'll work on a mutable copy of the MST
T = nx.Graph()
T.add_nodes_from(mst.nodes(data=True))
for u, v, d in mst.edges(data=True):
    T.add_edge(u, v, **d)

anonymized = set()

# Sort MST edges by original (physical) distance descending (longest first)
edges_sorted = sorted(mst.edges(data=True), key=lambda e: e[2].get('orig_dist', e[2]['weight']), reverse=True)

# We'll iterate and try to remove edges
for u, v, edata in edges_sorted:
    if not T.has_edge(u, v):
        continue
    # try removing edge
    T.remove_edge(u, v)
    comps = list(nx.connected_components(T))
    # compute total units per component
    comp_units = [sum(T.nodes[n]['est_units'] for n in comp) for comp in comps]
    # check if any component violates the MIN_UNITS rule
    violation = False
    small_comp_nodes = None
    for comp, units in zip(comps, comp_units):
        if units < MIN_UNITS:
            violation = True
            small_comp_nodes = comp
            break
    if not violation:
        # removal is fine; continue
        continue
    # There is a small component. Decide by max-distance rule or restore edge.
    orig_d = edata.get('orig_dist', edata['weight'])
    node_a_area = G.nodes[u]['gross_floor_area']
    node_b_area = G.nodes[v]['gross_floor_area']
    M_allowed = DISTANCE_FACTOR_M * 2 * (math.sqrt(node_a_area/math.pi) + math.sqrt(node_b_area/math.pi))
    if orig_d > M_allowed:
        # mark nodes in the small component as anonymized and remove them from T
        for n in list(small_comp_nodes):
            anonymized.add(n)
            if T.has_node(n):
                T.remove_node(n)
    else:
        # restore edge
        T.add_edge(u, v, **edata)

# Final groups are connected components of T
groups = list(nx.connected_components(T))

# compute group attributes (units, total demand, floor area, specific demand)
group_data = []
for gid, comp in enumerate(groups, start=1):
    comp_nodes = list(comp)
    total_units = sum(buildings[n]['est_units'] for n in comp_nodes)
    total_floor_area = sum(buildings[n]['gross_floor_area'] for n in comp_nodes)
    total_kwh = sum(buildings[n]['total_kwh_a'] for n in comp_nodes)
    specific_kwh_m2a = total_kwh / total_floor_area if total_floor_area>0 else 0
    group_data.append({
        'group_id': f'G{gid}',
        'nodes': comp_nodes,
        'total_units': total_units,
        'total_floor_area': total_floor_area,
        'total_kwh_a': total_kwh,
        'specific_kwh_m2a': specific_kwh_m2a
    })

# --- 5) Create aggregate polygon (two-buffer approximation) ---
def aggregate_polygon_for_group(comp_nodes):
    # union footprints
    polys = [buildings[n]['geom'] for n in comp_nodes]
    union = unary_union(polys)
    if union.is_empty:
        return None
    # compute half of the longest original edge among node pairs in the group
    max_d = 0.0
    for i in range(len(comp_nodes)):
        for j in range(i+1, len(comp_nodes)):
            n1, n2 = comp_nodes[i], comp_nodes[j]
            d = euclidean(centroids[n1], centroids[n2])
            if d > max_d:
                max_d = d
    if max_d == 0.0:
        buf_dist = 2.0
    else:
        buf_dist = 0.5 * max_d * BUFFER_TRIAL_MULTIPLIER
    # cap the buffer to avoid huge geometry in demo
    buf_dist = min(buf_dist, MAX_BUFFER_CAP)
    buf_out = union.buffer(buf_dist, join_style=2)
    # buffer inwards
    buf_in = buf_out.buffer(-buf_dist)
    # if result is empty or invalid, fallback to convex hull
    if buf_in.is_empty or not buf_in.is_valid:
        return union.convex_hull
    return buf_in

for g in group_data:
    g['geom'] = aggregate_polygon_for_group(g['nodes'])

# --- 6) Visualization ---
# compute data extent for axis limits
all_x = []
all_y = []
for b in buildings.values():
    for x,y in b['geom'].exterior.coords:
        all_x.append(x); all_y.append(y)
if all_x and all_y:
    xmin, xmax = min(all_x)-10, max(all_x)+10
    ymin, ymax = min(all_y)-10, max(all_y)+10
else:
    xmin, xmax, ymin, ymax = -10, 10, -10, 10

fig, ax = plt.subplots(figsize=(10,6))
ax.set_aspect('equal', 'box')

# define color bins for specific demand: 5 classes at 50 kWh/m2*a as in paper (bins: <50,50-100,...)
bins = [0,50,100,150,200,1000]
cmap = plt.get_cmap('RdYlGn_r')  # reversed so green = low demand, red = high
def color_for_value(v):
    # normalize into 0..1 via bins
    for i in range(len(bins)-1):
        if bins[i] <= v < bins[i+1]:
            frac = i / (len(bins)-2)  # 0..1 over 4 intervals
            return cmap(frac)
    return cmap(1.0)

# plot raw buildings (light grey)
for bid, b in buildings.items():
    poly = b['geom']
    coords = list(poly.exterior.coords)
    patch = MplPolygon(coords, closed=True, facecolor='lightgrey', edgecolor='k', alpha=0.5, linewidth=0.5)
    ax.add_patch(patch)
    cx, cy = poly.centroid.x, poly.centroid.y
    ax.text(cx, cy, bid, fontsize=7, ha='center', va='center')

# plot group geometries outlines with color = specific demand, linewidth ~ total demand
for g in group_data:
    geom = g['geom']
    if geom is None:
        continue
    color = color_for_value(g['specific_kwh_m2a'])
    # linewidth scaling
    lw = 0.7 + math.log1p(g['total_kwh_a'])/10.0
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        patch = MplPolygon(coords, closed=True, facecolor='none', edgecolor=color, linewidth=lw)
        ax.add_patch(patch)
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            coords = list(part.exterior.coords)
            patch = MplPolygon(coords, closed=True, facecolor='none', edgecolor=color, linewidth=lw)
            ax.add_patch(patch)

# highlight anonymized buildings
for n in anonymized:
    poly = buildings[n]['geom']
    coords = list(poly.exterior.coords)
    patch = MplPolygon(coords, closed=True, facecolor='none', edgecolor='black', hatch='////', linewidth=1.0)
    ax.add_patch(patch)
    cx, cy = poly.centroid.x, poly.centroid.y
    ax.text(cx, cy-1.0, 'ANON', fontsize=6, ha='center', va='center', color='black')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title('Aggregated building groups (outline color = specific kWh/m²·a, line width ~ total kWh/a)')
ax.axis('off')

# Print summary info
print("Buildings (dummy):", len(buildings))
print("Anonymized nodes:", anonymized)
print("Number of groups produced:", len(group_data))
print("\nGroup summary:")
for g in group_data:
    print(f" {g['group_id']}: nodes={g['nodes']}, total_units={g['total_units']}, specific_kWh/m2a={g['specific_kwh_m2a']:.1f}, total_kWh/a={g['total_kwh_a']:.0f}")

plt.show()
