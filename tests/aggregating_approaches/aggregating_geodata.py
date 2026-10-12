"""
aggregating_geodata.py
======================
Applies the MST-based building-aggregation approach from aggregating_method.py
to real geodata in the Gebaeude_NRW_Endstruktur format.

Input
-----
- Gebäude-GeoPackage  : buildings with heat demand, load profiles, floor area
- Flurstücke-GeoPackage: parcels used to assign a shared plot-ID per building

Output
------
- matplotlib visualisation (colour = specific heat demand)
- optional GeoPackage with aggregated group polygons

Column mapping (Gebäude layer)
-------------------------------
  new_ID          → building identifier
  Nutzung / typ   → use classification, used for unit estimation
  NF [m²]         → net floor area (proxy for number of residential units)
  WB [kWh/a]      → total annual heat demand
  Spez_WB [kWh/a*m²] → specific annual heat demand
  Lastprofil      → load-profile label
  geometry        → building footprint (Polygon / MultiPolygon)
"""

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Default paths  (edit here or pass as CLI arguments)
# ---------------------------------------------------------------------------
DEFAULT_BUILDINGS_PATH = str(
    Path(__file__).parents[2]
    / "testdata" / "kea_bw" / "Gebaeude_NRW_Endstruktur_KEA.gpkg"
)
DEFAULT_PARCELS_PATH = str(
    Path(__file__).parents[2]
    / "testdata" / "kea_bw" / "Flurstücke_Geometrie_NRW_Endstruktur.gpkg"
)
DEFAULT_OUTPUT_PATH = str(
    Path(__file__).parents[2]
    / "testdata" / "output_results" / "aggregated_groups.gpkg"
)

# ---------------------------------------------------------------------------
# Parameters (same semantics as aggregating_method.py)
# ---------------------------------------------------------------------------
MIN_UNITS = 10            # k-anonymity threshold (minimum dwelling units per group)
MIN_BUILDINGS = 5     # optional: minimum number of buildings per group (None = disabled)
PLOT_REWEIGHT = 0.05     # edge-weight multiplier when two buildings share a parcel
DISTANCE_FACTOR_M = 0.8  # max-distance factor for anonymisation decision
MAX_BUFFER_CAP = 40.0    # safety cap [m] for buffer when creating group polygons

# Residential keywords in the "Nutzung" column (case-insensitive substring match)
RESIDENTIAL_KEYWORDS = [
    "wohnhaus", "wohnen", "mfh", "efh", "wohngebäude",
    "mehrfamilienhaus", "einfamilienhaus", "reihenhaus",
]
# Approximate net floor area per residential unit [m²]
M2_PER_UNIT = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_use(row: dict) -> str:
    """Return 'residential' or 'non-residential' based on Nutzung / typ."""
    nutzung = str(row.get("Nutzung") or "").lower()
    typ = str(row.get("typ") or "").lower()
    for kw in RESIDENTIAL_KEYWORDS:
        if kw in nutzung or kw in typ:
            return "residential"
    return "non-residential"


def estimate_units(row: dict) -> int:
    """Estimate the number of dwelling units from NF [m²] and use class."""
    use = classify_use(row)
    if use == "residential":
        nf = row.get("NF [m²]") or 0.0
        return max(1, round(float(nf) / M2_PER_UNIT))
    return 1


def aggregate_polygon_for_group(
    footprints: list[Polygon], centroids_list
) -> Polygon | MultiPolygon | None:
    """Build an aggregate polygon via double-buffer (same as aggregating_method.py)."""
    if not footprints:
        return None
    union = unary_union(footprints)
    if union.is_empty:
        return None

    max_d = 0.0
    for i in range(len(centroids_list)):
        for j in range(i + 1, len(centroids_list)):
            c1, c2 = centroids_list[i], centroids_list[j]
            d = math.hypot(c1.x - c2.x, c1.y - c2.y)
            if d > max_d:
                max_d = d

    buf_dist = min(0.5 * max_d, MAX_BUFFER_CAP) if max_d > 0 else 2.0
    buf_out = union.buffer(buf_dist, join_style=2)
    buf_in = buf_out.buffer(-buf_dist)
    if buf_in.is_empty or not buf_in.is_valid:
        return union.convex_hull
    return buf_in


def color_for_value(v: float, bins=None, cmap=None):
    """Map a specific kWh/m²·a value to a colour from the given colourmap."""
    if bins is None:
        bins = [0, 50, 100, 150, 200, 1_000]
    if cmap is None:
        cmap = plt.get_cmap("RdYlGn_r")
    for i in range(len(bins) - 1):
        if bins[i] <= v < bins[i + 1]:
            frac = i / (len(bins) - 2)
            return cmap(frac)
    return cmap(1.0)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(
    buildings_path: str,
    parcels_path: str | None,
    output_path: str | None,
    save: bool = False,
    show_plot: bool = True,
    min_buildings: int | None = MIN_BUILDINGS,
) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    min_buildings : int or None
        If set, a group must contain at least this many buildings in addition
        to the MIN_UNITS threshold.  None disables the criterion.
    """
    # ------------------------------------------------------------------
    # 1. Load buildings
    # ------------------------------------------------------------------
    print(f"Loading buildings: {buildings_path}")
    gdf = gpd.read_file(buildings_path)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf.reset_index(drop=True)

    # Ensure projected CRS (metric coordinates needed for distance calc)
    if gdf.crs is None or gdf.crs.is_geographic:
        print("  Warning: no projected CRS detected – results may be inaccurate.")
    print(f"  {len(gdf)} buildings loaded (CRS: {gdf.crs})")

    # Representative point for each geometry (robust for MultiPolygon)
    gdf["_centroid"] = gdf.geometry.representative_point()

    # ------------------------------------------------------------------
    # 2. Assign plot-ID via spatial join with parcels (optional)
    # ------------------------------------------------------------------
    if parcels_path:
        print(f"Loading parcels: {parcels_path}")
        parcels = gpd.read_file(
            parcels_path,
            layer="Flurstücke_Geometrie_NRW_Endstruktur",
        )
        parcels = parcels[parcels.geometry.notna()].copy()
        parcels = parcels.to_crs(gdf.crs)

        joined = gpd.sjoin(
            gdf[["_centroid", "geometry"]].set_geometry("_centroid"),
            parcels[["gml_id", "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        gdf["plot_id"] = joined["gml_id"].reindex(gdf.index).fillna("unknown")
        print(f"  Plot IDs assigned ({gdf['plot_id'].nunique()} unique parcels)")
    else:
        gdf["plot_id"] = "unknown"

    # ------------------------------------------------------------------
    # 3. Derive per-building attributes
    # ------------------------------------------------------------------
    records = gdf.to_dict(orient="records")
    buildings: dict[int, dict] = {}
    for i, rec in enumerate(records):
        row_id = rec.get("new_ID", i)
        geom = rec["geometry"]
        centroid = gdf["_centroid"].iloc[i]

        # representative footprint for unary_union (take largest polygon if Multi)
        if isinstance(geom, MultiPolygon):
            footprint = max(geom.geoms, key=lambda g: g.area)
        elif isinstance(geom, Polygon):
            footprint = geom
        else:
            footprint = geom.convex_hull

        use = classify_use(rec)
        units = estimate_units(rec)
        total_kwh = float(rec.get("WB [kWh/a]") or 0.0)
        spez_kwh = float(rec.get("Spez_WB [kWh/a*m²]") or 0.0)
        nf = float(rec.get("NF [m²]") or 0.0)

        buildings[i] = {
            "db_id": row_id,
            "geom": footprint,
            "centroid": centroid,
            "plot_id": rec.get("plot_id", "unknown"),
            "use": use,
            "est_units": units,
            "gross_floor_area": nf,
            "total_kwh_a": total_kwh,
            "specific_kwh_m2a": spez_kwh,
            "Nutzung": rec.get("Nutzung"),
            "typ": rec.get("typ"),
            "Lastprofil": rec.get("Lastprofil"),
        }

    ids = list(buildings.keys())
    print(f"  {len(ids)} valid buildings prepared for aggregation")

    # ------------------------------------------------------------------
    # 4. Build complete graph with plot-reweighted distances
    # ------------------------------------------------------------------
    print("Building distance graph …")
    G = nx.Graph()
    for bid in ids:
        G.add_node(bid, **buildings[bid])

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            c1 = buildings[a]["centroid"]
            c2 = buildings[b]["centroid"]
            dist = math.hypot(c1.x - c2.x, c1.y - c2.y)
            same_plot = (
                buildings[a]["plot_id"] == buildings[b]["plot_id"]
                and buildings[a]["plot_id"] != "unknown"
            )
            weight = dist * (PLOT_REWEIGHT if same_plot else 1.0)
            G.add_edge(a, b, weight=weight, orig_dist=dist)

    # ------------------------------------------------------------------
    # 5. Minimum Spanning Tree + iterative edge removal
    # ------------------------------------------------------------------
    print("Computing MST and splitting groups …")
    mst = nx.minimum_spanning_tree(G, weight="weight")

    T = nx.Graph()
    T.add_nodes_from(mst.nodes(data=True))
    T.add_edges_from(mst.edges(data=True))

    anonymized: set[int] = set()

    edges_sorted = sorted(
        mst.edges(data=True),
        key=lambda e: e[2].get("orig_dist", e[2]["weight"]),
        reverse=True,
    )

    for u, v, edata in edges_sorted:
        if not T.has_edge(u, v):
            continue
        T.remove_edge(u, v)
        comps = list(nx.connected_components(T))
        comp_units = [
            sum(T.nodes[n]["est_units"] for n in comp) for comp in comps
        ]
        violation = False
        small_comp = None
        for comp, units in zip(comps, comp_units):
            units_violation = units < MIN_UNITS
            buildings_violation = (
                min_buildings is not None and len(comp) < min_buildings
            )
            if units_violation or buildings_violation:
                violation = True
                small_comp = comp
                break

        if not violation:
            continue

        orig_d = edata.get("orig_dist", edata["weight"])
        area_u = G.nodes[u]["gross_floor_area"]
        area_v = G.nodes[v]["gross_floor_area"]
        m_allowed = DISTANCE_FACTOR_M * 2 * (
            math.sqrt(area_u / math.pi) + math.sqrt(area_v / math.pi)
        ) if (area_u > 0 and area_v > 0) else 0.0

        if orig_d > m_allowed:
            for n in list(small_comp):
                anonymized.add(n)
                if T.has_node(n):
                    T.remove_node(n)
        else:
            T.add_edge(u, v, **edata)

    groups = list(nx.connected_components(T))
    print(f"  {len(groups)} groups formed, {len(anonymized)} buildings anonymised")

    # ------------------------------------------------------------------
    # 6. Compute group attributes and aggregate polygon
    # ------------------------------------------------------------------
    group_records = []
    for gid, comp in enumerate(groups, start=1):
        comp_nodes = list(comp)
        total_units = sum(buildings[n]["est_units"] for n in comp_nodes)
        total_floor = sum(buildings[n]["gross_floor_area"] for n in comp_nodes)
        total_kwh = sum(buildings[n]["total_kwh_a"] for n in comp_nodes)
        spez = total_kwh / total_floor if total_floor > 0 else 0.0
        footprints = [buildings[n]["geom"] for n in comp_nodes]
        centroids_list = [buildings[n]["centroid"] for n in comp_nodes]
        profiles = list({buildings[n]["Lastprofil"] for n in comp_nodes} - {None})
        nutzungen = list({buildings[n]["Nutzung"] for n in comp_nodes} - {None})

        group_records.append(
            {
                "group_id": f"G{gid}",
                "n_buildings": len(comp_nodes),
                "total_units": total_units,
                "total_floor_area_m2": total_floor,
                "total_heat_demand_kWh_a": total_kwh,
                "specific_heat_demand_kWh_m2a": spez,
                "Lastprofile": ", ".join(profiles),
                "Nutzungen": ", ".join(nutzungen),
                "geometry": aggregate_polygon_for_group(footprints, centroids_list),
            }
        )

    group_gdf = gpd.GeoDataFrame(group_records, crs=gdf.crs)

    # ------------------------------------------------------------------
    # 7. Save output
    # ------------------------------------------------------------------
    if save and output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        group_gdf.to_file(output_path, driver="GPKG")
        print(f"Saved aggregated groups to: {output_path}")

    # ------------------------------------------------------------------
    # 8. Visualisation
    # ------------------------------------------------------------------
    if show_plot:
        print("Rendering visualisation …")
        cmap = plt.get_cmap("RdYlGn_r")
        bins = [0, 50, 100, 150, 200, 1_000]

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_aspect("equal", "box")

        # Raw building footprints (light grey)
        for bid, b in buildings.items():
            geom = b["geom"]
            if isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            else:
                polys = [geom]
            for poly in polys:
                patch = MplPolygon(
                    list(poly.exterior.coords),
                    closed=True,
                    facecolor="lightgrey",
                    edgecolor="k",
                    alpha=0.5,
                    linewidth=0.4,
                )
                ax.add_patch(patch)

        # Group polygons (coloured outline)
        for _, row in group_gdf.iterrows():
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                continue
            col = color_for_value(row["specific_heat_demand_kWh_m2a"], bins, cmap)
            lw = 0.8 + math.log1p(row["total_heat_demand_kWh_a"]) / 12.0
            if isinstance(geom, MultiPolygon):
                parts = list(geom.geoms)
            else:
                parts = [geom]
            for part in parts:
                if isinstance(part, Polygon):
                    patch = MplPolygon(
                        list(part.exterior.coords),
                        closed=True,
                        facecolor="none",
                        edgecolor=col,
                        linewidth=lw,
                    )
                    ax.add_patch(patch)

        # Anonymised buildings (hatched)
        for n in anonymized:
            geom = buildings[n]["geom"]
            if isinstance(geom, MultiPolygon):
                parts = list(geom.geoms)
            else:
                parts = [geom]
            for part in parts:
                patch = MplPolygon(
                    list(part.exterior.coords),
                    closed=True,
                    facecolor="none",
                    edgecolor="black",
                    hatch="////",
                    linewidth=1.0,
                )
                ax.add_patch(patch)

        # Colourbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=bins[0], vmax=bins[-2]),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Specific heat demand [kWh/m²·a]", shrink=0.6)

        # Auto-fit axes
        all_x = [b["centroid"].x for b in buildings.values()]
        all_y = [b["centroid"].y for b in buildings.values()]
        margin = 50
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        ax.set_title(
            f"Aggregated building groups  |  "
            f"{len(groups)} groups  |  {len(anonymized)} anonymised  |  "
            f"outline colour = specific heat demand [kWh/m²·a]"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Summary
    print("\nGroup summary:")
    for rec in group_records:
        print(
            f"  {rec['group_id']}: "
            f"buildings={rec['n_buildings']}, "
            f"units={rec['total_units']}, "
            f"spez={rec['specific_heat_demand_kWh_m2a']:.1f} kWh/m²a, "
            f"total={rec['total_heat_demand_kWh_a']:.0f} kWh/a, "
            f"profiles=[{rec['Lastprofile']}]"
        )

    return group_gdf


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MST-based building aggregation for Gebaeude_NRW_Endstruktur data"
    )
    parser.add_argument(
        "--buildings", default=DEFAULT_BUILDINGS_PATH,
        help="Path to the Gebäude GeoPackage"
    )
    parser.add_argument(
        "--parcels", default=DEFAULT_PARCELS_PATH,
        help="Path to the Flurstücke GeoPackage (optional, for plot-ID assignment)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH,
        help="Output path for aggregated groups GeoPackage"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save aggregated groups to output GeoPackage"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip visualisation"
    )
    parser.add_argument(
        "--min-buildings", type=int, default=None,
        help="Minimum number of buildings per group (default: disabled). "
             "Example: --min-buildings 3"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        buildings_path=args.buildings,
        parcels_path=args.parcels,
        output_path=args.output,
        save=args.save,
        show_plot=not args.no_plot,
        min_buildings=args.min_buildings,
    )
