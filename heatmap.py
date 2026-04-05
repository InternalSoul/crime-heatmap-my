from __future__ import annotations

from pathlib import Path
import json
import random
import time
import numpy as np

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_CANDIDATES = [BASE_DIR / "crime_district.csv", BASE_DIR / "crime_data.csv"]
OUTPUT_FILE = BASE_DIR / "crime_heatmap.html"
PREVIEW_FILE = BASE_DIR / "crime_heatmap_preview.html"
MAIN_PAGE_FILE = BASE_DIR / "index.html"
MANUAL_COORD_OVERRIDES_FILE = BASE_DIR / "manual_coordinate_overrides.csv"

DISTRICT_NORMALIZATION = {
  "Cameron Highland": "Cameron Highlands",
}

DISTRICT_DIRECTIONAL_SUFFIXES = (
  " utara",
  " selatan",
  " timur",
  " barat",
  " north",
  " south",
  " east",
  " west",
)


def collapse_directional_district_name(name: str) -> str:
    cleaned = " ".join(str(name).split()).strip()
    lowered = cleaned.lower()
    for suffix in DISTRICT_DIRECTIONAL_SUFFIXES:
        if lowered.endswith(suffix):
            return cleaned[: -len(suffix)].strip()
    return cleaned


def load_data() -> pd.DataFrame:
    for csv_path in CSV_CANDIDATES:
        if csv_path.exists():
            return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No crime CSV found. Looked for: {', '.join(str(p.name) for p in CSV_CANDIDATES)}"
    )


def load_coordinates() -> pd.DataFrame:
    coord_file = BASE_DIR / "district_coordinates.csv"
    if coord_file.exists():
        coords_df = pd.read_csv(coord_file)
        if MANUAL_COORD_OVERRIDES_FILE.exists():
            overrides_df = pd.read_csv(MANUAL_COORD_OVERRIDES_FILE)
            required = {"state", "district", "latitude", "longitude"}
            if required.issubset(overrides_df.columns):
                overrides_df = overrides_df.dropna(subset=["state", "district", "latitude", "longitude"])
                coords_df = pd.concat([coords_df, overrides_df], ignore_index=True)
                coords_df = coords_df.drop_duplicates(subset=["state", "district"], keep="last")
        return coords_df
    raise FileNotFoundError(f"Coordinates file not found at {coord_file}")


def normalize_district_names(df: pd.DataFrame) -> pd.DataFrame:
  normalized = df.copy()
  normalized["district"] = (
    normalized["district"]
    .astype(str)
    .str.strip()
    .replace(DISTRICT_NORMALIZATION)
    .apply(collapse_directional_district_name)
  )
  return normalized


def build_heatmap_from_coordinates(df: pd.DataFrame, coords_df: pd.DataFrame) -> list[dict]:
    """Build heatmap with grid-split crime distribution for street-level accuracy."""
    required_columns = {"state", "district", "category", "type", "date", "crimes"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain state/district/category/type/date/crimes columns.")

    working = normalize_district_names(
      df.dropna(subset=["state", "district", "category", "type", "date", "crimes"]).copy()
    )
    coords_df = normalize_district_names(coords_df)
    coords_df["latitude"] = pd.to_numeric(coords_df["latitude"], errors="coerce")
    coords_df["longitude"] = pd.to_numeric(coords_df["longitude"], errors="coerce")
    coords_df = coords_df.dropna(subset=["state", "district", "latitude", "longitude"])
    coords_df = (
      coords_df.groupby(["state", "district"], as_index=False)[["latitude", "longitude"]]
      .mean()
    )
    working["year"] = pd.to_datetime(working["date"], errors="coerce").dt.year
    working = working.dropna(subset=["year"])

    aggregated = (
        working.groupby(["state", "district", "category", "type", "year"], as_index=False)["crimes"]
        .sum()
    )

    merged = aggregated.merge(coords_df, on=["state", "district"], how="left")

    heat_data: list[dict] = []
    
    BASE_SPREAD_DEG = 0.0035
    MAX_SPREAD_DEG = 0.02
    POINTS_PER_HUNDRED_CRIMES = 3

    for _, row in merged.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue

        center_lat = float(row["latitude"])
        center_lon = float(row["longitude"])
        crimes = int(row["crimes"])
        
        num_points = max(1, int(np.ceil(crimes / 100.0 * POINTS_PER_HUNDRED_CRIMES)))
        crime_scale = min(3.8, 1.0 + float(np.log1p(crimes)) / 2.1)
        spread = min(MAX_SPREAD_DEG, BASE_SPREAD_DEG * crime_scale)
        micro_cluster_count = min(6, max(1, int(np.ceil(np.log10(crimes + 1)))))

        micro_clusters = []
        for _ in range(micro_cluster_count):
            micro_clusters.append(
                (
                    center_lat + random.gauss(0.0, spread * 0.6),
                    center_lon + random.gauss(0.0, spread * 0.6),
                )
            )

        for _ in range(num_points):
            cluster_lat, cluster_lon = random.choice(micro_clusters)
            local_spread = max(0.0012, spread * random.uniform(0.45, 1.2))

            point_lat = cluster_lat + random.gauss(0.0, local_spread)
            point_lon = cluster_lon + random.gauss(0.0, local_spread)
            point_crimes = crimes / num_points
            point_intensity = min(
                2.5,
                max(0.45, random.uniform(0.75, 1.25) * (1.0 + float(np.log10(crimes + 1)) / 4.0)),
            )

            heat_data.append(
                {
                    "lat": point_lat,
                    "lon": point_lon,
                    "intensity": round(float(point_intensity), 4),
                    "crimes": point_crimes,
                    "state": str(row["state"]),
                    "district": str(row["district"]),
                    "category": str(row["category"]),
                    "type": str(row["type"]),
                    "year": int(row["year"]),
                }
            )

    print(f"🎯 Street-Level Grid: {len(heat_data)} points generated (grid ~2km/cell)")
    return heat_data


def summarize_heat_data(heat_data: list[dict]) -> dict:
    total_crimes = sum(float(p.get("crimes", 0.0)) for p in heat_data)
    states = sorted({str(p.get("state", "")) for p in heat_data if p.get("state") and str(p.get("state", "")).lower() not in ["malaysia", "all"]})
    districts = sorted({str(p.get("district", "")) for p in heat_data if p.get("district") and str(p.get("district", "")).lower() not in ["all"]})
    types = sorted({str(p.get("type", "")) for p in heat_data if p.get("type") and str(p.get("type", "")).lower() not in ["all"]})
    years = sorted({int(p.get("year")) for p in heat_data if p.get("year") is not None})

    by_state: dict[str, float] = {}
    by_type: dict[str, float] = {}
    by_year: dict[int, float] = {}
    by_category: dict[str, float] = {}
    by_district: dict[str, float] = {}

    for p in heat_data:
      state = str(p.get("state", "Unknown"))
      crime_type = str(p.get("type", "Unknown"))
      category = str(p.get("category", "Unknown"))
      district = str(p.get("district", "Unknown"))
      year = int(p.get("year", 0))
      crimes = float(p.get("crimes", 0.0))

      if state.lower() not in ["malaysia", "all"]:
        by_state[state] = by_state.get(state, 0.0) + crimes

      if crime_type.lower() not in ["all"]:
        by_type[crime_type] = by_type.get(crime_type, 0.0) + crimes

      if district.lower() not in ["all"]:
        by_district[district] = by_district.get(district, 0.0) + crimes
      if category.lower() not in ["all"]:
        by_category[category] = by_category.get(category, 0.0) + crimes
      by_year[year] = by_year.get(year, 0.0) + crimes

    # Get all items sorted by value (descending)
    all_states = sorted(by_state.items(), key=lambda x: x[1], reverse=True)
    all_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)
    all_districts = sorted(by_district.items(), key=lambda x: x[1], reverse=True)
    all_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
    
    # Keep top lists for cards
    top_states = all_states[:8]
    top_types = all_types[:6]
    top_districts = all_districts[:6]
    top_categories = all_categories[:6]

    return {
        "total_crimes": int(round(total_crimes)),
        "total_points": len(heat_data),
        "states_count": len(states),
        "districts_count": len(districts),
        "types_count": len(types),
        "years_count": len(years),
        "top_state": top_states[0][0] if top_states else "N/A",
        "top_type": top_types[0][0].replace("_", " ").title() if top_types else "N/A",
        "top_states": [{"name": k, "value": int(round(v))} for k, v in top_states],
        "top_types": [{"name": k.replace("_", " ").title(), "value": int(round(v))} for k, v in top_types],
        "top_districts": [{"name": k, "value": int(round(v))} for k, v in top_districts],
        "top_categories": [{"name": k.replace("_", " ").title(), "value": int(round(v))} for k, v in top_categories],
        "all_states": [{"name": k, "value": int(round(v))} for k, v in all_states],
        "all_types": [{"name": k.replace("_", " ").title(), "value": int(round(v))} for k, v in all_types],
        "all_districts": [{"name": k, "value": int(round(v))} for k, v in all_districts],
        "yearly": [{"year": int(k), "value": int(round(v))} for k, v in sorted(by_year.items())],
    }


def write_html_map(heat_data: list[dict], output_file: Path) -> None:
    points_js = json.dumps(heat_data, separators=(",", ":"))

    states = sorted({p["state"] for p in heat_data if p.get("state") and p["state"].lower() not in ["malaysia", "all"]})
    districts_by_state: dict[str, list[str]] = {
        state: sorted({p["district"] for p in heat_data if p.get("state") == state and p.get("district") and p["district"].lower() != "all"})
        for state in states
    }
    years = sorted({int(p["year"]) for p in heat_data if p.get("year") is not None})
    types = sorted({p["type"] for p in heat_data if p.get("type") and p["type"].lower() != "all"})

    # Initial state stats fallback content (shown even before JS filtering runs)
    state_totals: dict[str, float] = {}
    for p in heat_data:
      state = str(p.get("state", "")).strip()
      if not state or state.lower() in ["all", "malaysia"]:
        continue
      state_totals[state] = state_totals.get(state, 0.0) + float(p.get("crimes", 0.0))
    initial_state_rows = "".join(
      f'<div class="stat-row"><span class="stat-label">{state}</span><span class="stat-value">{int(round(total))}</span></div>'
      for state, total in sorted(state_totals.items(), key=lambda kv: kv[1], reverse=True)
    )

    # Build centroids per district for generated boundary polygons
    district_centroid_bins: dict[tuple[str, str], dict[str, float]] = {}
    for p in heat_data:
      state = str(p.get("state", "")).strip()
      district = str(p.get("district", "")).strip()
      if not state or not district:
        continue
      if state.lower() in ["malaysia", "all"] or district.lower() == "all":
        continue
      key = (state, district)
      prev = district_centroid_bins.get(key)
      if prev is None:
        district_centroid_bins[key] = {
          "lat_sum": float(p.get("lat", 0.0)),
          "lon_sum": float(p.get("lon", 0.0)),
          "count": 1.0,
        }
        continue
      prev["lat_sum"] += float(p.get("lat", 0.0))
      prev["lon_sum"] += float(p.get("lon", 0.0))
      prev["count"] += 1.0

    district_centroids = []
    for (state, district), bucket in district_centroid_bins.items():
      count = max(1.0, bucket["count"])
      district_centroids.append(
        {
          "state": state,
          "district": district,
          "lat": round(bucket["lat_sum"] / count, 6),
          "lon": round(bucket["lon_sum"] / count, 6),
        }
      )

    colors = {
      "murder": "#c1121f",
      "rape": "#008a2e",
      "causing_injury": "#ff9f1c",
      "robbery_gang_armed": "#7a1e48",
      "robbery_gang_unarmed": "#d97b00",
      "robbery_solo_armed": "#f4e04d",
      "robbery_solo_unarmed": "#3b82f6",
      "break_in": "#003f88",
      "theft_other": "#2a9d8f",
      "theft_vehicle_lorry": "#6f4e37",
      "theft_vehicle_motorcar": "#6a4c93",
      "theft_vehicle_motorcycle": "#00a6d6",
    }

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Crime Heatmap - Street Level</title>
  <link rel="manifest" href="manifest.webmanifest" />
  <meta name="theme-color" content="#0f766e" />
  <meta name="mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-title" content="CrimeMap MY" />
  <link rel="icon" href="icon.svg" type="image/svg+xml" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {
      --panel-bg: rgba(255, 255, 255, 0.9);
      --panel-border: rgba(255, 255, 255, 0.42);
      --text-main: #1f2937;
      --text-soft: #5b6472;
      --accent: #0f766e;
      --accent-strong: #0f5a54;
      --chip-bg: #eef6ff;
      --chip-text: #15589b;
    }
    html, body, #map { height: 100%; margin: 0; font-family: 'Manrope', 'Segoe UI', Tahoma, sans-serif; }
    body {
      background: radial-gradient(circle at 14% 18%, #d7f0ff 0%, #eef4ff 38%, #f3f6ff 70%, #f8fbff 100%);
      color: var(--text-main);
    }
    #map {
      filter: saturate(1.03) contrast(1.02);
    }
    .title-group {
      position: absolute;
      z-index: 1150;
      top: 16px;
      left: 50%;
      transform: translateX(-50%);
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .app-title {
      padding: 9px 14px;
      border-radius: 999px;
      border: 1px solid var(--panel-border);
      background: linear-gradient(120deg, rgba(255,255,255,0.94), rgba(241,250,255,0.92));
      box-shadow: 0 10px 26px rgba(7, 37, 62, 0.16);
      font-weight: 800;
      font-size: 13px;
      letter-spacing: 0.3px;
      color: #163a52;
    }
    .home-btn {
      display: inline-flex;
      align-items: center;
      text-decoration: none;
      font-size: 18px;
      font-weight: 800;
      line-height: 1;
      color: #0b3c5d;
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.44);
      border-radius: 999px;
      width: 34px;
      height: 34px;
      justify-content: center;
      box-shadow: 0 8px 20px rgba(5, 31, 45, 0.2);
    }
    .home-btn:hover { background: #f4fbff; }
    .install-btn {
      display: none;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      font-size: 12px;
      font-weight: 800;
      color: #ffffff;
      background: linear-gradient(135deg, #0f766e, #15589b);
      border: 1px solid rgba(255, 255, 255, 0.44);
      border-radius: 999px;
      height: 34px;
      padding: 0 12px;
      box-shadow: 0 8px 20px rgba(5, 31, 45, 0.2);
      cursor: pointer;
    }
    .install-btn:hover { filter: brightness(1.05); }
    .install-toast {
      position: fixed;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%) translateY(16px);
      background: rgba(15, 118, 110, 0.96);
      color: #fff;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.2px;
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease, transform 0.2s ease;
      z-index: 1400;
    }
    .install-toast.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
    .mobile-toolbar {
      display: none;
      position: absolute;
      z-index: 1310;
      top: 58px;
      left: 8px;
      right: 8px;
      gap: 6px;
    }
    .mobile-toolbar button {
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.35);
      background: rgba(15, 118, 110, 0.9);
      color: #fff;
      font-size: 11px;
      font-weight: 800;
      padding: 7px 10px;
      box-shadow: 0 6px 14px rgba(0, 0, 0, 0.15);
    }
    .mobile-toolbar button.active {
      background: rgba(21, 88, 155, 0.96);
      border-color: rgba(255, 255, 255, 0.55);
    }
    .mobile-panel-backdrop {
      display: none;
      position: absolute;
      inset: 0;
      z-index: 1200;
      background: rgba(5, 20, 34, 0.28);
      pointer-events: none;
    }
    .mobile-panel-backdrop.show {
      display: block;
      pointer-events: auto;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
      position: sticky;
      top: 0;
      z-index: 3;
      background: rgba(255,255,255,0.94);
      backdrop-filter: blur(6px);
      padding-bottom: 6px;
      border-bottom: 1px solid #e5ebf2;
    }
    .panel-head h3 {
      margin: 0;
      border-bottom: none;
      padding-bottom: 0;
    }
    .panel-close {
      width: auto;
      min-width: 28px;
      height: 28px;
      border-radius: 999px;
      border: 1px solid #d4deea;
      background: #f8fbff;
      color: #334155;
      font-size: 14px;
      line-height: 1;
      padding: 0 8px;
      margin: 0;
      flex: 0 0 auto;
    }
    .panel-close:hover {
      background: #eef5ff;
    }
    .rotate-overlay {
      display: none;
      position: fixed;
      inset: 0;
      z-index: 2000;
      background: linear-gradient(180deg, rgba(9, 26, 43, 0.95), rgba(10, 39, 66, 0.95));
      color: #fff;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 24px;
      font-family: 'Manrope', 'Segoe UI', Tahoma, sans-serif;
    }
    .rotate-overlay .rotate-box {
      max-width: 320px;
    }
    .rotate-overlay .rotate-title {
      font-size: 20px;
      font-weight: 800;
      margin-bottom: 8px;
    }
    .rotate-overlay .rotate-text {
      font-size: 13px;
      opacity: 0.95;
      line-height: 1.45;
    }
    .panel {
      position: absolute;
      z-index: 1300;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(7px);
      -webkit-backdrop-filter: blur(7px);
      padding: 12px;
      border-radius: 14px;
      box-shadow: 0 12px 30px rgba(0, 27, 46, 0.16);
      overflow-y: auto;
      font-size: 13px;
      overscroll-behavior: contain;
      -webkit-overflow-scrolling: touch;
      touch-action: pan-y;
      pointer-events: auto;
    }
    .filters-panel { top: 16px; left: 16px; width: 360px; max-height: 66vh; overflow: hidden; }
    .stats-panel { top: 16px; right: 16px; width: 290px; max-height: 42vh; overflow: hidden; }
    .legend-panel { bottom: 16px; right: 16px; width: 290px; max-height: 34vh; overflow: hidden; }
    .summary-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0 0 10px 0;
    }
    .summary-chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: var(--chip-bg);
      border: 1px solid #d4e8ff;
      color: var(--chip-text);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }
    .data-toggle {
      position: absolute;
      left: 12px;
      bottom: 12px;
      z-index: 1000;
      border: 1px solid rgba(255, 255, 255, 0.35);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      width: auto;
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: #fff;
      box-shadow: 0 10px 24px rgba(5, 31, 45, 0.25);
    }
    .data-drawer {
      position: absolute;
      z-index: 1200;
      left: 16px;
      right: 16px;
      bottom: 16px;
      max-height: 42vh;
      display: none;
      background: rgba(255,255,255,0.96);
      border: 1px solid var(--panel-border);
      padding: 12px;
      border-radius: 14px;
      box-shadow: 0 12px 30px rgba(0, 22, 40, 0.2);
    }
    .data-drawer.open { display: block; }
    h3 {
      margin: 0 0 10px 0;
      font-size: 16px;
      font-weight: 800;
      border-bottom: 2px solid #e5ebf2;
      padding-bottom: 8px;
      color: #1c3148;
    }
    .field { margin-bottom: 10px; }
    .label { display: block; margin-bottom: 6px; font-weight: 700; color: #38495d; }
    .field-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .field-head .label { margin-bottom: 0; }
    .head-actions { display: inline-flex; gap: 6px; margin-left: 10px; }
    select, button { width: 100%; padding: 6px; border: 1px solid #bbb; border-radius: 4px; }
    button { background: #f0f0f0; cursor: pointer; font-weight: 700; margin-right: 4px; flex: 1; }
    button:hover { background: #e0e0e0; }
    .mini-btn {
      width: auto;
      min-width: 46px;
      padding: 3px 8px;
      margin-right: 0;
      border-radius: 999px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      font-size: 11px;
      line-height: 1.2;
      color: #334155;
      flex: 0 0 auto;
      transition: background 0.2s ease, border-color 0.2s ease;
    }
    .mini-btn:hover { background: #eef5ff; border-color: #9ec5ff; }
    .checkbox-list {
      border: 1px solid #d5dde8;
      border-radius: 10px;
      max-height: 130px;
      overflow-y: auto;
      padding: 6px;
      background: #fff;
      overscroll-behavior: contain;
      -webkit-overflow-scrolling: touch;
      touch-action: pan-y;
    }
    .checkbox-item {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: #334155;
      padding: 3px 4px;
      border-radius: 6px;
    }
    .checkbox-item:hover { background: #f3f8ff; }
    .checkbox-item input { margin-right: 6px; }
    #filters-content {
      max-height: calc(66vh - 112px);
      overflow-y: auto;
      padding-right: 4px;
      overscroll-behavior: contain;
      -webkit-overflow-scrolling: touch;
      touch-action: pan-y;
    }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 6px; padding: 3px 2px; }
    .stat-label { font-weight: 700; color: #48566a; }
    .stat-value { color: #0f766e; font-weight: 800; }
    #stats-content {
      max-height: calc(42vh - 58px);
      overflow-y: auto;
      padding-right: 4px;
      overscroll-behavior: contain;
      -webkit-overflow-scrolling: touch;
      touch-action: pan-y;
    }
    #legend-content {
      max-height: calc(34vh - 58px);
      overflow-y: auto;
      padding-right: 4px;
      overscroll-behavior: contain;
      -webkit-overflow-scrolling: touch;
      touch-action: pan-y;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
      padding: 3px 4px;
      border-radius: 6px;
    }
    .legend-item:hover { background: #f6fbff; }
    .legend-swatch {
      width: 16px;
      height: 16px;
      min-width: 16px;
      border-radius: 50%;
      border: 1px solid rgba(0, 0, 0, 0.2);
    }
    .close-drawer-btn {
      background: #cf3e2d;
      color: white;
      border: none;
      padding: 6px 10px;
      cursor: pointer;
      border-radius: 8px;
      float: right;
      width: auto;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    table th { background: #edf3fb; padding: 6px; text-align: left; font-weight: 800; color: #334155; }
    table td { padding: 6px; border-bottom: 1px solid #e5e9ef; color: #374151; }
    @media (max-width: 900px) and (orientation: landscape) {
      .title-group { top: 8px; }
      .app-title { font-size: 11px; padding: 7px 11px; }
      .home-btn { width: 28px; height: 28px; font-size: 14px; }
      .install-btn { height: 28px; font-size: 10px; padding: 0 9px; }
      .mobile-toolbar {
        display: flex;
        top: auto;
        bottom: 10px;
        justify-content: center;
      }
      .mobile-toolbar button {
        width: auto;
        min-width: 64px;
        padding: 6px 10px;
      }
      .panel { display: none; }
      .panel.mobile-open { display: block; }
      .filters-panel,
      .stats-panel,
      .legend-panel {
        top: 44px;
        right: 10px;
        left: auto;
        width: min(40vw, 310px);
        max-height: calc(100vh - 98px);
        padding: 10px;
      }
      .data-toggle { bottom: 50px; left: 10px; }
      .data-drawer { left: 10px; right: 10px; bottom: 50px; max-height: 44vh; }
      #filters-content { max-height: calc(100vh - 190px); }
      #stats-content { max-height: calc(100vh - 160px); }
      #legend-content { max-height: calc(100vh - 160px); }
      .checkbox-list { max-height: 92px; }
      .checkbox-item { font-size: 11px; margin-bottom: 2px; padding: 2px 3px; }
      .summary-chip { font-size: 10px; padding: 3px 6px; }
      h3 { font-size: 14px; margin-bottom: 8px; padding-bottom: 6px; }
      .panel-close { display: inline-flex; align-items: center; justify-content: center; }
    }
    @media (max-width: 900px) and (orientation: portrait) {
      .rotate-overlay { display: flex; }
      .panel,
      .mobile-toolbar,
      .data-toggle,
      .data-drawer {
        display: none !important;
      }
    }
    body.preview-mode .app-title,
    body.preview-mode .home-btn,
    body.preview-mode .install-btn,
    body.preview-mode .panel,
    body.preview-mode .data-toggle,
    body.preview-mode .data-drawer {
      display: none !important;
    }
    body.preview-mode #map {
      filter: none;
    }
    body.preview-mode {
      background: transparent;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="title-group">
    <a class="home-btn" href="index.html" title="Return to main page" aria-label="Return to main page">↩</a>
    <div class="app-title">Malaysia Crime Heatmap Explorer</div>
    <button id="install-app-btn" class="install-btn" type="button">Install App</button>
  </div>
  <div class="mobile-toolbar" id="mobile-toolbar">
    <button type="button" data-panel="filters-panel" onclick="toggleMobilePanel('filters-panel')">Filters</button>
    <button type="button" data-panel="stats-panel" onclick="toggleMobilePanel('stats-panel')">Stats</button>
    <button type="button" data-panel="legend-panel" onclick="toggleMobilePanel('legend-panel')">Legend</button>
  </div>
  <div id="mobile-panel-backdrop" class="mobile-panel-backdrop" onclick="closeMobilePanels()"></div>
  <div class="rotate-overlay" id="rotate-overlay">
    <div class="rotate-box">
      <div class="rotate-title">Rotate Your Phone</div>
      <div class="rotate-text">For best heatmap visibility and filter controls, use landscape orientation.</div>
    </div>
  </div>
  <div id="install-toast" class="install-toast">App installed successfully</div>

  <div id="filters-panel" class="panel filters-panel">
    <div class="panel-head">
      <h3>Filters</h3>
      <button type="button" class="panel-close" onclick="closeMobilePanels()">×</button>
    </div>
    <div id="summary-strip" class="summary-strip"></div>
    <div id="filters-content">
    <div class="field">
      <div class="field-head"><label class="label">State</label><div class="head-actions"><button class="mini-btn" onclick="selectAllStates()">All</button><button class="mini-btn" onclick="deselectAllStates()">None</button></div></div>
      <div id="state-select" class="checkbox-list"></div>
    </div>
    <div class="field">
      <div class="field-head"><label class="label">District</label><div class="head-actions"><button class="mini-btn" onclick="selectAllDistricts()">All</button><button class="mini-btn" onclick="deselectAllDistricts()">None</button></div></div>
      <div id="district-select" class="checkbox-list"></div>
    </div>
    <div class="field">
      <div class="field-head"><label class="label">Year</label><div class="head-actions"><button class="mini-btn" onclick="selectAllYears()">All</button><button class="mini-btn" onclick="deselectAllYears()">None</button></div></div>
      <div id="year-select" class="checkbox-list"></div>
    </div>
    <div class="field">
      <div class="field-head"><label class="label">Crime Type</label><div class="head-actions"><button class="mini-btn" onclick="selectAllTypes()">All</button><button class="mini-btn" onclick="deselectAllTypes()">None</button></div></div>
      <div id="type-select" class="checkbox-list"></div>
    </div>
    <div class="field" style="margin-top: 8px;">
      <label class="checkbox-item" style="display:flex; align-items:center; gap:8px; margin:0;">
        <input id="boundary-toggle" type="checkbox" checked>
        Show district boundaries
      </label>
    </div>
    <button onclick="resetFilters()" style="width: 100%; margin-top: 10px; background: #ff9800; color: white;">Reset All</button>
    </div>
  </div>

  <div id="stats-panel" class="panel stats-panel">
    <div class="panel-head">
      <h3 id="stats-title">Crime by State</h3>
      <button type="button" class="panel-close" onclick="closeMobilePanels()">×</button>
    </div>
    <div id="stats-content">__INITIAL_STATE_ROWS__</div>
  </div>

  <div id="legend-panel" class="panel legend-panel">
    <div class="panel-head">
      <h3>Legend</h3>
      <button type="button" class="panel-close" onclick="closeMobilePanels()">×</button>
    </div>
    <div id="legend-content"></div>
  </div>

  <button id="data-toggle-btn" class="data-toggle" onclick="toggleDataPanel()">Show Table</button>
  
  <div class="data-drawer">
    <button class="close-drawer-btn" onclick="toggleDataPanel()">Close</button>
    <h3>Filtered Crime Data</h3>
    <div id="data-table-container" style="overflow-y: auto; max-height: 38vh;"></div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3-delaunay@6/d3-delaunay.min.js"></script>
  <script>
    const heatPoints = __POINTS__;
    const colorMap = __COLORS__;
    const statesList = __STATES__;
    const districtsByState = __DISTRICTS_BY_STATE__;
    const yearsList = __YEARS__;
    const typesList = __TYPES__;
    const districtCentroids = __DISTRICT_CENTROIDS__;
    const MAX_POINTS_PER_TYPE = 5500;
    const crimeRecords = buildCrimeRecords(heatPoints);

    let map;
    let heatLayers = [];
    let districtBoundaryLayer = null;
    let pointMarkerLayer = null;
    let pointRenderer = null;
    let lastRenderedPoints = [];
    let lastRenderedUnifiedMode = false;
    let showDistrictBoundaries = true;
    const isPreviewMode = applyPreviewMode();
    let selected = { state: new Set(statesList), district: new Set(), year: new Set(yearsList), type: new Set(typesList) };

    const els = {
      stateSelect: document.getElementById('state-select'),
      districtSelect: document.getElementById('district-select'),
      yearSelect: document.getElementById('year-select'),
      typeSelect: document.getElementById('type-select'),
      statsTitle: document.getElementById('stats-title'),
      statsContent: document.getElementById('stats-content'),
      legendContent: document.getElementById('legend-content'),
      summaryStrip: document.getElementById('summary-strip'),
      dataTableContainer: document.getElementById('data-table-container'),
      dataDrawer: document.querySelector('.data-drawer'),
      dataToggleBtn: document.getElementById('data-toggle-btn'),
      mobilePanelBackdrop: document.getElementById('mobile-panel-backdrop'),
      mobileToolbar: document.getElementById('mobile-toolbar'),
    };

    function initMap() {
      const initialView = isPreviewMode ? [3.95, 109.2] : [4.2105, 101.6964];
      const initialZoom = isPreviewMode ? 5 : 6;
      map = L.map('map', {
        preferCanvas: true,
        zoomAnimation: false,
        fadeAnimation: false,
        markerZoomAnimation: false,
      }).setView(initialView, initialZoom);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap',
        maxZoom: 19,
      }).addTo(map);
      if (isPreviewMode) {
        map.dragging.disable();
        map.scrollWheelZoom.disable();
        map.doubleClickZoom.disable();
        map.boxZoom.disable();
        map.keyboard.disable();
      }

      map.createPane('districtBoundaries');
      map.getPane('districtBoundaries').style.zIndex = 520;
      map.createPane('pointMarkers');
      map.getPane('pointMarkers').style.zIndex = 420;
      pointRenderer = L.canvas({ padding: 0.2 });

      renderHeatmap();
      renderDistrictBoundaries();
    }

    function lockMapInteractionsUnderPanels() {
      const blockers = [
        document.getElementById('filters-panel'),
        document.getElementById('stats-panel'),
        document.getElementById('legend-panel'),
        document.querySelector('.data-drawer'),
        document.getElementById('mobile-toolbar'),
      ].filter(Boolean);

      blockers.forEach(el => {
        L.DomEvent.disableClickPropagation(el);
        L.DomEvent.disableScrollPropagation(el);
        ['touchstart', 'touchmove', 'touchend', 'pointerdown', 'pointermove', 'mousedown', 'wheel'].forEach(evt => {
          L.DomEvent.on(el, evt, L.DomEvent.stopPropagation);
        });
      });
    }

    function applyPreviewMode() {
      const params = new URLSearchParams(window.location.search);
      const preview = params.get('preview') === '1';
      if (preview) document.body.classList.add('preview-mode');
      return preview;
    }

    function populateSelects() {
      renderCheckboxGroup(els.stateSelect, statesList, selected.state, onStateChange, v => v);
      renderCheckboxGroup(els.yearSelect, yearsList, selected.year, onFilterChange, v => String(v));
      renderCheckboxGroup(els.typeSelect, typesList, selected.type, onFilterChange, v => v.replace(/_/g, ' ').toUpperCase());
      renderLegend();
      onStateChange();
    }

    function renderCheckboxGroup(container, values, selectedSet, onChange, labelFormatter) {
      container.innerHTML = '';
      values.forEach(value => {
        const label = document.createElement('label');
        label.className = 'checkbox-item';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = String(value);
        cb.checked = selectedSet.has(value);
        cb.addEventListener('change', onChange);
        label.appendChild(cb);
        label.appendChild(document.createTextNode(labelFormatter(value)));
        container.appendChild(label);
      });
    }

    function getChecked(container) {
      return [...container.querySelectorAll('input[type="checkbox"]:checked')].map(cb => cb.value);
    }

    function onStateChange() {
      selected.state = new Set(getChecked(els.stateSelect));
      const availableDistricts = new Set();
      heatPoints.forEach(p => {
        if (selected.state.has(p.state) && String(p.district || '').toLowerCase() !== 'all') {
          availableDistricts.add(p.district);
        }
      });
      selected.district = new Set([...availableDistricts]);
      renderCheckboxGroup(els.districtSelect, [...availableDistricts].sort(), selected.district, onFilterChange, v => v);
      onFilterChange();
    }

    function onFilterChange() {
      selected.district = new Set(getChecked(els.districtSelect));
      selected.year = new Set(getChecked(els.yearSelect).map(v => parseInt(v)));
      selected.type = new Set(getChecked(els.typeSelect));
      updateSummaryStrip();
      const filteredPoints = getFilteredPoints();
      const filteredRecords = getFilteredRecords();
      renderStats(filteredPoints);
      renderDataTable(filteredRecords);
      try {
        renderHeatmap(filteredPoints);
        renderDistrictBoundaries();
      } catch (err) {
        console.error('Map render error:', err);
      }
    }

    function buildCrimeRecords(points) {
      const grouped = new Map();
      points.forEach(p => {
        const state = String(p.state || '').trim();
        const district = String(p.district || '').trim();
        const type = String(p.type || '').trim();
        const year = parseInt(p.year, 10);
        const crimes = Number(p.crimes || 0);
        if (!state || !district || !type || Number.isNaN(year) || !Number.isFinite(crimes)) return;
        const key = `${state}|${district}|${type}|${year}`;
        grouped.set(key, (grouped.get(key) || 0) + crimes);
      });

      return Array.from(grouped.entries()).map(([key, crimes]) => {
        const [state, district, type, year] = key.split('|');
        return {
          state,
          district,
          type,
          year: parseInt(year, 10),
          crimes,
        };
      });
    }

    function getFilteredRecords() {
      return crimeRecords.filter(r =>
        selected.state.has(r.state) &&
        selected.district.has(r.district) &&
        String(r.district || '').toLowerCase() !== 'all' &&
        selected.year.has(r.year) &&
        selected.type.has(r.type)
      );
    }

    function getBoundaryCentroids() {
      return districtCentroids.filter(d =>
        selected.state.has(d.state) && selected.district.has(d.district)
      );
    }

    function getCentroidBounds(points, padding = 0.8) {
      let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
      points.forEach(p => {
        if (p.lat < minLat) minLat = p.lat;
        if (p.lat > maxLat) maxLat = p.lat;
        if (p.lon < minLon) minLon = p.lon;
        if (p.lon > maxLon) maxLon = p.lon;
      });

      // Keep cells bounded over Malaysia while giving enough room near edges.
      const latPad = Math.max(0.45, (maxLat - minLat) * 0.2 + padding * 0.08);
      const lonPad = Math.max(0.6, (maxLon - minLon) * 0.2 + padding * 0.1);
      return [minLon - lonPad, minLat - latPad, maxLon + lonPad, maxLat + latPad];
    }

    function renderDistrictBoundaries() {
      if (districtBoundaryLayer) {
        map.removeLayer(districtBoundaryLayer);
        districtBoundaryLayer = null;
      }
      if (isPreviewMode || !showDistrictBoundaries) return;

      const boundaryPoints = getBoundaryCentroids();
      if (boundaryPoints.length < 3) return;

      const extent = getCentroidBounds(boundaryPoints);
      const delaunay = d3.Delaunay.from(boundaryPoints, p => p.lon, p => p.lat);
      const voronoi = delaunay.voronoi(extent);

      districtBoundaryLayer = L.layerGroup();
      boundaryPoints.forEach((point, i) => {
        const poly = voronoi.cellPolygon(i);
        if (!poly || poly.length < 3) return;
        const latLngs = poly.map(([lon, lat]) => [lat, lon]);
        L.polygon(latLngs, {
          pane: 'districtBoundaries',
          color: '#0b3c5d',
          weight: 1.6,
          opacity: 0.92,
          fillColor: '#93c5fd',
          fillOpacity: 0.07,
          dashArray: '5 4',
          interactive: false,
          smoothFactor: 1.0,
        }).addTo(districtBoundaryLayer);
      });

      districtBoundaryLayer.addTo(map);
    }

    function onBoundaryToggleChange(event) {
      showDistrictBoundaries = Boolean(event && event.target && event.target.checked);
      if (!showDistrictBoundaries) {
        if (districtBoundaryLayer) {
          map.removeLayer(districtBoundaryLayer);
          districtBoundaryLayer = null;
        }
        return;
      }
      renderDistrictBoundaries();
    }

    function updateSummaryStrip() {
      const chips = [
        { label: 'States', value: `${selected.state.size}/${statesList.length}` },
        { label: 'Districts', value: `${selected.district.size}` },
        { label: 'Years', value: `${selected.year.size}/${yearsList.length}` },
        { label: 'Types', value: `${selected.type.size}/${typesList.length}` },
      ];
      els.summaryStrip.innerHTML = chips
        .map(chip => `<span class="summary-chip">${chip.label}: ${chip.value}</span>`)
        .join('');
    }

    function getFilteredPoints() {
      return heatPoints.filter(p =>
        selected.state.has(p.state) &&
        selected.district.has(p.district) &&
        String(p.district || '').toLowerCase() !== 'all' &&
        selected.year.has(p.year) &&
        selected.type.has(p.type)
      );
    }

    function renderHeatmap(filteredInput = null) {
      const filtered = filteredInput || getFilteredPoints();
      heatLayers.forEach(layer => map.removeLayer(layer));
      heatLayers = [];
      if (pointMarkerLayer) {
        map.removeLayer(pointMarkerLayer);
        pointMarkerLayer = null;
      }

      const zoom = map.getZoom();
      const heatStyle = getHeatLayerStyle(zoom);
      const selectedTypeCount = selected.type.size;
      const useUnifiedLayer = selectedTypeCount >= 6 || selectedTypeCount === typesList.length;
      lastRenderedPoints = filtered;
      lastRenderedUnifiedMode = useUnifiedLayer;

      if (useUnifiedLayer) {
        const unifiedPoints = filtered.map(p => [p.lat, p.lon, p.intensity ?? 1.0]);
        const optimizedUnified = optimizeHeatPoints(unifiedPoints, zoom);
        if (optimizedUnified.length) {
          const unifiedLayer = L.heatLayer(optimizedUnified, {
            radius: heatStyle.radius,
            blur: heatStyle.blur,
            minOpacity: heatStyle.minOpacity,
            maxZoom: 12,
            max: heatStyle.max,
            gradient: {
              0.0: 'rgba(0,0,0,0)',
              0.35: 'rgba(56,189,248,0.5)',
              0.6: 'rgba(15,118,110,0.72)',
              0.82: 'rgba(245,158,11,0.9)',
              1.0: 'rgba(220,38,38,1.0)',
            },
          }).addTo(map);
          heatLayers.push(unifiedLayer);
        }
        renderPointMarkers(filtered, true);
        return;
      }

      const pointsByType = {};
      filtered.forEach(p => {
        if (!pointsByType[p.type]) pointsByType[p.type] = [];
        pointsByType[p.type].push([p.lat, p.lon, p.intensity ?? 1.0]);
      });

      Object.entries(pointsByType).forEach(([crimeType, points]) => {
        if (!points.length) return;
        const typeColor = colorMap[crimeType] || '#ff0000';
        const optimized = optimizeHeatPoints(points, zoom);
        const layer = L.heatLayer(optimized, {
          radius: heatStyle.radius,
          blur: heatStyle.blur,
          minOpacity: heatStyle.minOpacity,
          maxZoom: 12,
          max: heatStyle.max,
          gradient: {
            0.0: 'rgba(0,0,0,0)',
            0.45: hexToRgba(typeColor, heatStyle.alphaMid),
            0.7: hexToRgba(typeColor, heatStyle.alphaHigh),
            1.0: hexToRgba(typeColor, heatStyle.alphaPeak),
          },
        }).addTo(map);
        heatLayers.push(layer);
      });

      renderPointMarkers(filtered, false);
    }

    function getHeatLayerStyle(zoom) {
      if (zoom >= 14) {
        return {
          radius: 16,
          blur: 9,
          minOpacity: 0.2,
          max: 8,
          alphaMid: 0.6,
          alphaHigh: 0.8,
          alphaPeak: 1.0,
        };
      }
      if (zoom >= 12) {
        return {
          radius: 15,
          blur: 10,
          minOpacity: 0.18,
          max: 8,
          alphaMid: 0.52,
          alphaHigh: 0.72,
          alphaPeak: 0.95,
        };
      }
      if (zoom >= 10) {
        return {
          radius: 16,
          blur: 13,
          minOpacity: 0.14,
          max: 9,
          alphaMid: 0.42,
          alphaHigh: 0.62,
          alphaPeak: 0.86,
        };
      }
      if (zoom >= 8) {
        return {
          radius: 15,
          blur: 16,
          minOpacity: 0.11,
          max: 9,
          alphaMid: 0.26,
          alphaHigh: 0.4,
          alphaPeak: 0.56,
        };
      }
      return {
        radius: 22,
        blur: 24,
        minOpacity: 0.16,
        max: 7,
        alphaMid: 0.4,
        alphaHigh: 0.62,
        alphaPeak: 0.82,
      };
    }

    function renderPointMarkers(filteredPoints, unifiedMode) {
      if (isPreviewMode) return;

      const zoom = map.getZoom();
      if (zoom < 13 || !filteredPoints.length) return;

      const bounds = map.getBounds();
      const visiblePoints = filteredPoints.filter(p => bounds.contains([p.lat, p.lon]));
      if (!visiblePoints.length) return;

      // Keep sampled marker anchors stable across zoom levels.
      const maxMarkers = 3200;
      const step = Math.max(1, Math.ceil(visiblePoints.length / maxMarkers));
      pointMarkerLayer = L.layerGroup();

      for (let i = 0; i < visiblePoints.length; i += step) {
        const p = visiblePoints[i];
        const color = unifiedMode ? '#ef4444' : (colorMap[p.type] || '#ef4444');
        L.circleMarker([p.lat, p.lon], {
          pane: 'pointMarkers',
          renderer: pointRenderer,
          radius: zoom >= 15 ? 6.2 : 5.2,
          stroke: true,
          color: '#ffffff',
          weight: 1.4,
          fillColor: color,
          fillOpacity: 1.0,
          interactive: false,
        }).addTo(pointMarkerLayer);
      }

      pointMarkerLayer.addTo(map);
    }

    function optimizeHeatPoints(points, zoom) {
      const cellSize = getCellSizeForZoom(zoom);
      const bins = new Map();

      points.forEach(([lat, lon, intensity]) => {
        const keyLat = Math.round(lat / cellSize);
        const keyLon = Math.round(lon / cellSize);
        const key = `${keyLat}:${keyLon}`;
        const prev = bins.get(key);
        if (!prev) {
          bins.set(key, { latSum: lat, lonSum: lon, intensitySum: intensity, count: 1 });
          return;
        }
        prev.latSum += lat;
        prev.lonSum += lon;
        prev.intensitySum += intensity;
        prev.count += 1;
      });

      const aggregated = [];
      const zoomDamping = getIntensityDamping(zoom);
      bins.forEach(bin => {
        const avgLat = bin.latSum / bin.count;
        const avgLon = bin.lonSum / bin.count;
        const scaledIntensity = Math.min(
          2.4,
          Math.max(0.12, (bin.intensitySum / Math.sqrt(bin.count)) * zoomDamping)
        );
        aggregated.push([avgLat, avgLon, scaledIntensity]);
      });

      if (aggregated.length <= MAX_POINTS_PER_TYPE) return aggregated;

      const step = Math.ceil(aggregated.length / MAX_POINTS_PER_TYPE);
      const sampled = [];
      for (let i = 0; i < aggregated.length; i += step) {
        sampled.push(aggregated[i]);
      }
      return sampled;
    }

    function getIntensityDamping(zoom) {
      if (zoom >= 14) return 0.75;
      if (zoom >= 12) return 0.8;
      if (zoom >= 10) return 0.86;
      if (zoom >= 8) return 0.9;
      return 0.95;
    }

    function getCellSizeForZoom(zoom) {
      // Use a fixed aggregation grid so heat anchors do not shift while zooming.
      return 0.008;
    }

    function hexToRgba(hex, alpha) {
      const normalized = String(hex || '').trim().replace('#', '');
      if (!/^[0-9a-fA-F]{6}$/.test(normalized)) {
        return `rgba(255,0,0,${alpha})`;
      }
      const r = parseInt(normalized.substring(0, 2), 16);
      const g = parseInt(normalized.substring(2, 4), 16);
      const b = parseInt(normalized.substring(4, 6), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    function onMapZoomEnd() {
      renderHeatmap();
      renderDistrictBoundaries();
    }

    function onMapMoveEnd() {
      if (pointMarkerLayer) {
        map.removeLayer(pointMarkerLayer);
        pointMarkerLayer = null;
      }
      renderPointMarkers(lastRenderedPoints, lastRenderedUnifiedMode);
    }

    function renderStats(filteredPoints) {
      const selectedTypes = selected.type.size ? Array.from(selected.type) : typesList;
      if (selectedTypes.length === 1) {
        const rawLabel = selectedTypes[0].replace(/_/g, ' ');
        const label = rawLabel.charAt(0).toUpperCase() + rawLabel.slice(1);
        els.statsTitle.textContent = `${label} by State`;
      } else if (selectedTypes.length > 1 && selectedTypes.length < typesList.length) {
        els.statsTitle.textContent = 'Selected Crimes by State';
      } else {
        els.statsTitle.textContent = 'Crime by State';
      }
      const statsByState = {};
      filteredPoints.forEach(p => {
        const key = String(p.state || '').trim();
        if (!key) return;
        if (!statsByState[key]) statsByState[key] = 0;
        statsByState[key] += Number(p.crimes || 0);
      });
      const sorted = Object.entries(statsByState).sort((a, b) => b[1] - a[1]);
      if (!sorted.length) {
        els.statsContent.innerHTML = '<div class="stat-row"><span class="stat-label">No data</span><span class="stat-value">0</span></div>';
        return;
      }
      els.statsContent.innerHTML = sorted.map(([s, c]) =>
        `<div class="stat-row"><span class="stat-label">${s}</span><span class="stat-value">${Math.round(c)}</span></div>`
      ).join('');
    }

    function renderLegend() {
      els.legendContent.innerHTML = Object.entries(colorMap).map(([t, c]) => {
        const safeColor = String(c || '').trim();
        return `<div class="legend-item"><span class="legend-swatch" style="background-color: ${safeColor};"></span><span>${t.replace(/_/g, ' ')}</span></div>`;
      }).join('');
    }

    function renderDataTable(filteredRecords) {
      if (filteredRecords.length === 0) {
        els.dataTableContainer.innerHTML = '<p>No data for current filters.</p>';
        return;
      }
      const rows = filteredRecords
        .slice()
        .sort((a, b) => b.crimes - a.crimes)
        .slice(0, 100)
        .map(r =>
          `<tr><td>${r.state}</td><td>${r.district}</td><td>${r.type}</td><td>${r.year}</td><td>${Math.round(r.crimes)}</td></tr>`
        )
        .join('');
      els.dataTableContainer.innerHTML =
        `<table><thead><tr><th>State</th><th>District</th><th>Type</th><th>Year</th><th>Crimes</th></tr></thead><tbody>${rows}</tbody></table>
         <p style="font-size: 11px; color: #888;">Showing first 100 of ${filteredRecords.length} records.</p>`;
    }

    function toggleDataPanel() {
      if (window.innerWidth <= 900) {
        closeMobilePanels();
      }
      const isOpen = els.dataDrawer.classList.toggle('open');
      els.dataToggleBtn.textContent = isOpen ? 'Hide Table' : 'Show Table';
    }

    function closeMobilePanels() {
      if (window.innerWidth > 900) return;
      ['filters-panel', 'stats-panel', 'legend-panel'].forEach(id => {
        const panel = document.getElementById(id);
        if (panel) panel.classList.remove('mobile-open');
      });
      if (els.mobilePanelBackdrop) {
        els.mobilePanelBackdrop.classList.remove('show');
      }
      if (els.mobileToolbar) {
        els.mobileToolbar.querySelectorAll('button[data-panel]').forEach(btn => btn.classList.remove('active'));
      }
    }

    function toggleMobilePanel(panelId) {
      if (window.innerWidth > 900) return;
      const panel = document.getElementById(panelId);
      if (!panel) return;
      const wasOpen = panel.classList.contains('mobile-open');
      closeMobilePanels();
      if (!wasOpen) {
        panel.classList.add('mobile-open');
        if (els.mobilePanelBackdrop) {
          els.mobilePanelBackdrop.classList.add('show');
        }
        if (els.mobileToolbar) {
          const activeBtn = els.mobileToolbar.querySelector(`button[data-panel="${panelId}"]`);
          if (activeBtn) activeBtn.classList.add('active');
        }
      }
      els.dataDrawer.classList.remove('open');
      els.dataToggleBtn.textContent = 'Show Table';
    }

    function setAllChecked(container, checked) {
      [...container.querySelectorAll('input[type="checkbox"]')].forEach(cb => {
        cb.checked = checked;
      });
    }
    function selectAllStates() { setAllChecked(els.stateSelect, true); onStateChange(); }
    function deselectAllStates() { setAllChecked(els.stateSelect, false); onStateChange(); }
    function selectAllDistricts() { setAllChecked(els.districtSelect, true); onFilterChange(); }
    function deselectAllDistricts() { setAllChecked(els.districtSelect, false); onFilterChange(); }
    function selectAllYears() { setAllChecked(els.yearSelect, true); onFilterChange(); }
    function deselectAllYears() { setAllChecked(els.yearSelect, false); onFilterChange(); }
    function selectAllTypes() { setAllChecked(els.typeSelect, true); onFilterChange(); }
    function deselectAllTypes() { setAllChecked(els.typeSelect, false); onFilterChange(); }
    function resetFilters() { selectAllStates(); selectAllYears(); selectAllTypes(); onStateChange(); }

    initMap();
    lockMapInteractionsUnderPanels();
    map.on('zoomend', onMapZoomEnd);
    map.on('moveend', onMapMoveEnd);
    map.on('click', (event) => {
      const target = event && event.originalEvent ? event.originalEvent.target : null;
      if (target && target.closest && target.closest('#filters-panel, #stats-panel, #legend-panel, .data-drawer, #mobile-toolbar')) {
        return;
      }
      closeMobilePanels();
    });
    populateSelects();
    // Force consistent default selections on refresh so stats are always populated.
    resetFilters();
    // Safety render so the state panel always has content even if map layers fail.
    renderStats(getFilteredPoints());

    const boundaryToggle = document.getElementById('boundary-toggle');
    if (boundaryToggle) {
      boundaryToggle.checked = showDistrictBoundaries;
      boundaryToggle.addEventListener('change', onBoundaryToggleChange);
    }

    let deferredInstallPrompt = null;
    const installBtn = document.getElementById('install-app-btn');
    const installToast = document.getElementById('install-toast');

    function showInstallToast() {
      if (!installToast) return;
      installToast.classList.add('show');
      setTimeout(() => installToast.classList.remove('show'), 2200);
    }

    window.addEventListener('beforeinstallprompt', (event) => {
      event.preventDefault();
      deferredInstallPrompt = event;
      if (installBtn) installBtn.style.display = 'inline-flex';
    });

    window.addEventListener('appinstalled', () => {
      deferredInstallPrompt = null;
      if (installBtn) installBtn.style.display = 'none';
      showInstallToast();
    });

    if (installBtn) {
      installBtn.addEventListener('click', async () => {
        if (deferredInstallPrompt) {
          deferredInstallPrompt.prompt();
          await deferredInstallPrompt.userChoice;
          deferredInstallPrompt = null;
          installBtn.style.display = 'none';
          return;
        }
        alert('Use your browser menu and tap "Add to Home Screen" to install this app.');
      });
    }

    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('sw.js').then(reg => {
          reg.update();
        }).catch(() => {});
      });
    }
  </script>
</body>
</html>
"""

    html_content = html_content.replace("__POINTS__", points_js)
    html_content = html_content.replace("__COLORS__", json.dumps(colors, separators=(",", ":")))
    html_content = html_content.replace("__STATES__", json.dumps(states, separators=(",", ":")))
    html_content = html_content.replace("__DISTRICTS_BY_STATE__", json.dumps(districts_by_state, separators=(",", ":")))
    html_content = html_content.replace("__YEARS__", json.dumps(years, separators=(",", ":")))
    html_content = html_content.replace("__TYPES__", json.dumps(types, separators=(",", ":")))
    html_content = html_content.replace("__DISTRICT_CENTROIDS__", json.dumps(district_centroids, separators=(",", ":")))
    html_content = html_content.replace("__INITIAL_STATE_ROWS__", initial_state_rows)

    output_file.write_text(html_content, encoding="utf-8")


def build_preview_points(heat_data: list[dict], target_points: int = 12000) -> list[list[float]]:
    """Build a lightweight, aggregated points set for fast dashboard preview rendering."""
    if not heat_data:
        return []

    cell_size = 0.03
    bins: dict[tuple[int, int], dict[str, float]] = {}

    for p in heat_data:
        lat = float(p.get("lat", 0.0))
        lon = float(p.get("lon", 0.0))
        intensity = float(p.get("intensity", 1.0))
        key = (int(round(lat / cell_size)), int(round(lon / cell_size)))
        prev = bins.get(key)
        if prev is None:
            bins[key] = {"lat_sum": lat, "lon_sum": lon, "intensity_sum": intensity, "count": 1.0}
            continue
        prev["lat_sum"] += lat
        prev["lon_sum"] += lon
        prev["intensity_sum"] += intensity
        prev["count"] += 1.0

    aggregated: list[list[float]] = []
    for bucket in bins.values():
        count = max(1.0, bucket["count"])
        avg_lat = bucket["lat_sum"] / count
        avg_lon = bucket["lon_sum"] / count
        avg_intensity = bucket["intensity_sum"] / count
        scaled_intensity = min(2.4, max(0.28, avg_intensity * (1.0 + np.log1p(count) * 0.35)))
        aggregated.append([round(avg_lat, 6), round(avg_lon, 6), round(float(scaled_intensity), 4)])

    aggregated.sort(key=lambda point: point[2], reverse=True)
    if len(aggregated) > target_points:
        step = len(aggregated) / float(target_points)
        sampled = []
        idx = 0.0
        while int(idx) < len(aggregated) and len(sampled) < target_points:
            sampled.append(aggregated[int(idx)])
            idx += step
        return sampled

    return aggregated


def write_preview_map(heat_data: list[dict], output_file: Path) -> None:
    preview_points = build_preview_points(heat_data)
    points_js = json.dumps(preview_points, separators=(",", ":"))

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Crime Heatmap Preview</title>
  <link rel="manifest" href="manifest.webmanifest" />
  <meta name="theme-color" content="#0f766e" />
  <link rel="icon" href="icon.svg" type="image/svg+xml" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map { height: 100%; margin: 0; }
    body { background: #eef5ff; }
    #map { filter: saturate(1.02) contrast(1.01); }
    .preview-badge {
      position: absolute;
      z-index: 1000;
      top: 10px;
      left: 10px;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(15, 118, 110, 0.25);
      border-radius: 999px;
      padding: 5px 10px;
      font: 700 11px/1.2 'Segoe UI', Tahoma, sans-serif;
      color: #0f5a54;
      letter-spacing: 0.2px;
      user-select: none;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="preview-badge">Fast Preview</div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
  <script>
    const points = __PREVIEW_POINTS__;
    const map = L.map('map', {
      zoomControl: false,
      dragging: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      boxZoom: false,
      keyboard: false,
      attributionControl: false,
      touchZoom: false,
    }).setView([3.95, 109.2], 5);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 12,
    }).addTo(map);

    L.heatLayer(points, {
      radius: 22,
      blur: 26,
      minOpacity: 0.24,
      maxZoom: 10,
      max: 3.0,
      gradient: {
        0.0: 'rgba(0,0,0,0)',
        0.35: 'rgba(59,130,246,0.45)',
        0.6: 'rgba(15,118,110,0.65)',
        0.82: 'rgba(217,119,6,0.78)',
        1.0: 'rgba(193,18,31,0.92)',
      },
    }).addTo(map);

    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('sw.js').then(reg => {
          reg.update();
        }).catch(() => {});
      });
    }
  </script>
</body>
</html>
"""

    html_content = html_content.replace("__PREVIEW_POINTS__", points_js)
    output_file.write_text(html_content, encoding="utf-8")


def write_main_page(heat_data: list[dict], output_file: Path) -> None:
    summary = summarize_heat_data(heat_data)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Malaysia Crime Dashboard</title>
  <link rel="manifest" href="manifest.webmanifest" />
  <meta name="theme-color" content="#0f766e" />
  <meta name="mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-title" content="CrimeMap MY" />
  <link rel="icon" href="icon.svg" type="image/svg+xml" />
  <link rel="prefetch" href="crime_heatmap.html" as="document">
  <link rel="prefetch" href="crime_heatmap_preview.html" as="document">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg-primary: #f8fbff;
      --bg-secondary: #edf5ff;
      --card-bg: #ffffff;
      --border-color: #e3ebf5;
      --text-primary: #1f2b3a;
      --text-secondary: #5f6f85;
      --text-muted: #8b95a8;
      --accent-1: #0f766e;
      --accent-2: #15589b;
      --accent-3: #d97706;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; font-family: 'Manrope', 'Segoe UI', Tahoma, sans-serif; color: var(--text-primary); }
    body {
      min-height: 100vh;
      background: linear-gradient(135deg, #d9eeff 0%, var(--bg-secondary) 35%, var(--bg-primary) 100%);
      padding: 20px 24px;
    }
    
    .container { max-width: 1400px; margin: 0 auto; }
    
    .header {
      text-align: center;
      margin-bottom: 28px;
    }
    
    .header-content {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
    }
    
    .title {
      margin: 0;
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .subtitle {
      margin: 0;
      font-size: 14px;
      color: var(--text-secondary);
      font-weight: 500;
    }
    
    .install-app-btn {
      display: none;
      margin-top: 10px;
      border: 1px solid rgba(15, 118, 110, 0.35);
      background: linear-gradient(135deg, rgba(15,118,110,0.95), rgba(21,88,155,0.95));
      color: #fff;
      font-family: 'Manrope', 'Segoe UI', Tahoma, sans-serif;
      font-size: 12px;
      font-weight: 800;
      border-radius: 999px;
      padding: 9px 14px;
      cursor: pointer;
      box-shadow: 0 8px 16px rgba(7, 37, 62, 0.14);
    }
    .install-app-btn:hover {
      filter: brightness(1.05);
    }
    
    .install-toast {
      position: fixed;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%) translateY(16px);
      background: rgba(15, 118, 110, 0.96);
      color: #fff;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.2px;
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease, transform 0.2s ease;
      z-index: 1400;
    }
    .install-toast.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
    
    .stats-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      margin-bottom: 32px;
    }
    
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 18px 14px;
      box-shadow: 0 2px 8px rgba(7, 37, 62, 0.06);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      transition: all 0.3s ease;
      min-height: 120px;
      gap: 8px;
    }
    
    .stat-card:hover {
      box-shadow: 0 8px 16px rgba(7, 37, 62, 0.1);
      transform: translateY(-2px);
    }
    
    .stat-card .label {
      font-size: 10px;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      line-height: 1.3;
      word-break: break-word;
    }
    
    .stat-card .value {
      font-size: clamp(16px, 4vw, 24px);
      font-weight: 800;
      color: var(--accent-1);
      line-height: 1.2;
      word-break: break-word;
      overflow-wrap: break-word;
    }
    
    .stat-card.accent2 .value { color: var(--accent-2); }
    .stat-card.accent3 .value { color: var(--accent-3); }
    
    .section-title {
      font-size: 20px;
      font-weight: 800;
      margin: 28px 0 16px 0;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .section-title::before {
      content: '';
      width: 3px;
      height: 22px;
      background: var(--accent-1);
      border-radius: 2px;
    }
    
    .main-layout {
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      gap: 16px;
      margin-bottom: 28px;
      align-items: start;
    }
    
    .tabs-sidebar {
      display: flex;
      flex-direction: column;
      gap: 8px;
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 14px;
      height: fit-content;
      box-shadow: 0 10px 22px rgba(7, 37, 62, 0.08);
      position: sticky;
      top: 20px;
    }

    .tabs-sidebar-head {
      margin-bottom: 4px;
      padding: 6px 6px 8px;
      border-bottom: 1px solid var(--border-color);
    }

    .tabs-sidebar-title {
      margin: 0;
      font-size: 13px;
      color: var(--text-primary);
      font-weight: 800;
      letter-spacing: 0.2px;
    }

    .tabs-sidebar-subtitle {
      margin-top: 3px;
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 600;
    }
    
    .tab-btn {
      padding: 11px 12px;
      border: 1px solid #dbe7f2;
      border-radius: 10px;
      background: #f8fbff;
      color: var(--text-secondary);
      font-weight: 700;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      white-space: nowrap;
      text-align: left;
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .tab-emoji {
      font-size: 14px;
      line-height: 1;
    }

    .tab-label {
      line-height: 1;
    }
    
    .tab-btn::before {
      content: '';
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 0;
      background: linear-gradient(180deg, var(--accent-1), var(--accent-2));
      transition: width 0.2s ease;
    }

    .tab-btn:hover {
      background: #eef6ff;
      color: var(--text-primary);
      border-color: #cfe0ef;
    }
    
    .tab-btn.active {
      background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(21,88,155,0.14));
      border-color: rgba(15, 118, 110, 0.5);
      color: var(--accent-1);
      box-shadow: 0 4px 10px rgba(15, 118, 110, 0.12);
    }

    .tab-btn.active::before {
      width: 3px;
    }
    
    .tabs-content {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    .tab-panel {
      display: none;
      animation: fadeInTab 0.24s ease;
    }
    
    .tab-panel.active {
      display: block;
    }

    .tab-panel .section-title:first-child {
      margin-top: 0;
    }

    .section-emoji {
      font-size: 18px;
      line-height: 1;
      margin-right: 2px;
    }

    @keyframes fadeInTab {
      from {
        opacity: 0;
        transform: translateY(6px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .content-layout {
      display: grid;
      gap: 14px;
      grid-template-columns: 1fr 1fr;
      margin-bottom: 28px;
    }
    
    .content-layout.full { grid-template-columns: 1fr; }
    
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 4px 12px rgba(7, 37, 62, 0.08);
    }
    
    .panel h3 {
      margin: 0 0 14px 0;
      font-size: 16px;
      font-weight: 700;
      color: var(--text-primary);
      border-bottom: 2px solid var(--border-color);
      padding-bottom: 10px;
    }
    
    .chart-container {
      position: relative;
      height: 300px;
      margin-bottom: 8px;
    }
    
    .list-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px solid var(--border-color);
      font-size: 13px;
    }
    
    .list-item:last-child { border-bottom: none; }
    
    .item-name {
      color: var(--text-primary);
      font-weight: 600;
      flex: 1;
    }
    
    .item-value {
      color: var(--accent-2);
      font-weight: 800;
      min-width: 80px;
      text-align: right;
    }
    
    .progress-bar {
      width: 100%;
      height: 6px;
      background: var(--border-color);
      border-radius: 3px;
      overflow: hidden;
      margin-top: 6px;
    }
    
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
      border-radius: 3px;
      transition: width 0.3s ease;
    }
    
    .preview-container {
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      position: relative;
      height: 480px;
      background: linear-gradient(135deg, #f7fbff, #f0f7ff);
    }
    
    .preview-container iframe {
      width: 100%;
      height: 100%;
      border: 0;
      pointer-events: none;
    }
    
    .preview-overlay {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: end;
      justify-content: start;
      text-decoration: none;
      color: #fff;
      background: linear-gradient(180deg, rgba(0,0,0,0.02) 50%, rgba(0,0,0,0.5) 100%);
      padding: 18px;
      font-weight: 700;
      letter-spacing: 0.3px;
      font-size: 14px;
      transition: background 0.3s ease;
    }
    
    .preview-overlay:hover {
      background: linear-gradient(180deg, rgba(0,0,0,0.08) 50%, rgba(0,0,0,0.6) 100%);
    }
    
    .three-column-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-bottom: 28px;
    }
    
    .search-box {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      font-family: 'Manrope', sans-serif;
      font-size: 13px;
      margin-bottom: 12px;
      background: #f9fafb;
      transition: border-color 0.2s;
    }
    
    .search-box:focus {
      outline: none;
      border-color: var(--accent-1);
      background: #fff;
    }
    
    @media (max-width: 1200px) {
      .content-layout { grid-template-columns: 1fr; }
      .three-column-grid { grid-template-columns: repeat(2, 1fr); }
    }
    
    @media (max-width: 1024px) {
      .stats-grid { grid-template-columns: repeat(4, 1fr); }
    }
    
    @media (max-width: 768px) {
      body { padding: 16px 20px; }
      .title { font-size: 28px; }
      .stats-grid { grid-template-columns: repeat(3, 1fr); gap: 8px; }
      .stat-card { padding: 14px 10px; min-height: 100px; font-size: 12px; }
      .stat-card .label { font-size: 9px; }
      .stat-card .value { font-size: clamp(14px, 3.5vw, 20px); }
      .three-column-grid { grid-template-columns: 1fr; }
      .chart-container { height: 250px; }
      .preview-container { height: 360px; }
      .main-layout { grid-template-columns: 1fr; gap: 12px; }
      .tabs-sidebar {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        position: static;
      }
      .tabs-sidebar-head {
        grid-column: 1 / -1;
        margin-bottom: 2px;
      }
      .tabs-sidebar-title { font-size: 12px; }
      .tabs-sidebar-subtitle { font-size: 10px; }
      .tab-btn { padding: 9px 10px; font-size: 11px; min-width: 70px; justify-content: center; }
      .tab-emoji { font-size: 13px; }
    }
    
    @media (max-width: 480px) {
      .main-layout { grid-template-columns: 1fr; }
      .tabs-sidebar { gap: 6px; grid-template-columns: 1fr; }
      .tab-btn { padding: 8px; font-size: 10px; }
      .stats-grid { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <div class="header">
      <div class="header-content">
        <h1 class="title">Malaysia Crime Dashboard</h1>
        <p class="subtitle">Comprehensive insights into crime trends across Malaysia with interactive visualizations</p>
        <button id="install-app-btn" class="install-app-btn" type="button">Install App</button>
      </div>
    </div>
    <div id="install-toast" class="install-toast">App installed successfully</div>

    <!-- Top Stats -->
    <div class="stats-grid" id="stats-container"></div>

    <!-- Sidebar + Tabs Layout -->
    <div class="main-layout">
      <!-- Sidebar Navigation -->
      <div class="tabs-sidebar">
        <div class="tabs-sidebar-head">
          <p class="tabs-sidebar-title">Explore Dashboard</p>
          <div class="tabs-sidebar-subtitle">Choose a section view</div>
        </div>
        <button class="tab-btn active" onclick="switchTab('analysis', this)"><span class="tab-emoji">📊</span><span class="tab-label">Analysis</span></button>
        <button class="tab-btn" onclick="switchTab('data', this)"><span class="tab-emoji">📋</span><span class="tab-label">Data</span></button>
        <button class="tab-btn" onclick="switchTab('trends', this)"><span class="tab-emoji">📈</span><span class="tab-label">Trends</span></button>
        <button class="tab-btn" onclick="switchTab('map', this)"><span class="tab-emoji">🗺️</span><span class="tab-label">Map</span></button>
      </div>

      <!-- Tab Content -->
      <div class="tabs-content">
        
        <!-- Analysis Tab -->
        <div id="analysis-tab" class="tab-panel active">
          <div class="section-title"><span class="section-emoji">📊</span>Crime Distribution</div>
          <div class="content-layout">
            <div class="panel">
              <h3>Top States</h3>
              <div class="chart-container">
                <canvas id="statesChart"></canvas>
              </div>
            </div>
            <div class="panel">
              <h3>Crime Types</h3>
              <div class="chart-container">
                <canvas id="typesChart"></canvas>
              </div>
            </div>
          </div>

          <div class="section-title"><span class="section-emoji">🧩</span>All Crime Types</div>
          <div class="panel" style="margin-bottom: 28px;">
            <h3>Crime Type Distribution (All)</h3>
            <div style="position: relative; height: 400px;">
              <canvas id="allTypesChart"></canvas>
            </div>
          </div>
        </div>

        <!-- Trends Tab -->
        <div id="trends-tab" class="tab-panel">
          <div class="section-title"><span class="section-emoji">📈</span>Yearly Trends</div>
          <div class="panel" style="margin-bottom: 28px;">
            <h3>Crime Trend Over Years</h3>
            <div style="position: relative; height: 350px;">
              <canvas id="yearlyChart"></canvas>
            </div>
          </div>
        </div>

        <!-- Data Tab -->
        <div id="data-tab" class="tab-panel">
          <div class="section-title"><span class="section-emoji">🗂️</span>States & Districts</div>
          <div class="content-layout">
            <div class="panel">
              <h3>All States</h3>
              <input type="text" id="statesSearch" class="search-box" placeholder="Search states...">
              <div id="all-states-list" style="max-height: 500px; overflow-y: auto;"></div>
            </div>
            <div class="panel">
              <h3>All Districts</h3>
              <input type="text" id="districtsSearch" class="search-box" placeholder="Search districts...">
              <div id="all-districts-list" style="max-height: 500px; overflow-y: auto;"></div>
            </div>
          </div>

          <div class="section-title"><span class="section-emoji">🧾</span>Additional Details</div>
          <div class="three-column-grid">
            <div class="panel">
              <h3>Crime Categories</h3>
              <div id="categories-list"></div>
            </div>
            <div class="panel">
              <h3>Overview</h3>
              <div id="overview-list"></div>
            </div>
            <div class="panel">
              <h3>Year Statistics</h3>
              <div id="years-list"></div>
            </div>
          </div>
        </div>

        <!-- Map Tab -->
        <div id="map-tab" class="tab-panel">
          <div class="section-title"><span class="section-emoji">🗺️</span>Interactive Map</div>
          <div class="panel content-layout full" style="margin-bottom: 28px;">
            <div style="grid-column: 1 / -1;">
              <h3>Heatmap Preview</h3>
              <div class="preview-container">
                <iframe title="Heatmap Preview" src="crime_heatmap_preview.html" loading="lazy"></iframe>
                <a class="preview-overlay" href="crime_heatmap.html">→ Open interactive heatmap</a>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    const summary = __SUMMARY__;

    // Generate Stat Cards
    function renderStats() {
      const stats = [
        { label: 'Total Crimes', value: summary.total_crimes.toLocaleString(), class: '' },
        { label: 'Heat Points', value: summary.total_points.toLocaleString(), class: 'accent2' },
        { label: 'States', value: summary.states_count, class: 'accent3' },
        { label: 'Districts', value: summary.districts_count, class: '' },
        { label: 'Crime Types', value: summary.types_count, class: 'accent2' },
        { label: 'Years Covered', value: summary.years_count, class: 'accent3' },
        { label: 'Top State', value: summary.top_state, class: '' },
        { label: 'Top Crime', value: summary.top_type, class: 'accent2' },
      ];

      document.getElementById('stats-container').innerHTML = stats
        .map(stat => `<div class="stat-card ${stat.class}"><div class="label">${stat.label}</div><div class="value">${stat.value}</div></div>`)
        .join('');
    }

    // States Chart
    const statesCtx = document.getElementById('statesChart').getContext('2d');
    new Chart(statesCtx, {
      type: 'bar',
      data: {
        labels: summary.top_states.map(s => s.name),
        datasets: [{
          label: 'Total Crimes',
          data: summary.top_states.map(s => s.value),
          backgroundColor: [
            'rgba(15, 118, 110, 0.8)',
            'rgba(15, 118, 110, 0.65)',
            'rgba(21, 88, 155, 0.8)',
            'rgba(217, 119, 6, 0.8)',
            'rgba(21, 88, 155, 0.65)',
            'rgba(217, 119, 6, 0.65)',
            'rgba(15, 118, 110, 0.5)',
            'rgba(21, 88, 155, 0.5)',
          ],
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { font: { size: 11 } } },
          x: { ticks: { font: { size: 11 } }, grid: { display: false } }
        }
      }
    });

    // Crime Types Chart
    const typesCtx = document.getElementById('typesChart').getContext('2d');
    
    // Shorten long display names
    const shortenLabel = (label) => {
      if (label.includes('Theft Vehicle Motorcycle')) return 'Vehicle Motorcycle';
      if (label.includes('Theft Vehicle')) return 'Theft Vehicle';
      if (label.length > 20) return label.substring(0, 17) + '...';
      return label;
    };
    
    new Chart(typesCtx, {
      type: 'doughnut',
      data: {
        labels: summary.top_types.map(t => shortenLabel(t.name)),
        datasets: [{
          data: summary.top_types.map(t => t.value),
          backgroundColor: [
            'rgba(15, 118, 110, 0.85)',
            'rgba(21, 88, 155, 0.85)',
            'rgba(217, 119, 6, 0.85)',
            'rgba(239, 68, 68, 0.85)',
            'rgba(16, 185, 129, 0.85)',
            'rgba(168, 85, 247, 0.85)',
          ],
          borderColor: '#fff',
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { 
            position: 'bottom', 
            labels: { 
              font: { size: 10 }, 
              padding: 10, 
              usePointStyle: true,
              maxWidth: 80,
              boxWidth: 12
            } 
          }
        }
      }
    });

    // Yearly Trend Chart
    const yearlyCtx = document.getElementById('yearlyChart').getContext('2d');
    new Chart(yearlyCtx, {
      type: 'line',
      data: {
        labels: summary.yearly.map(y => y.year),
        datasets: [{
          label: 'Total Crimes',
          data: summary.yearly.map(y => y.value),
          borderColor: 'rgba(15, 118, 110, 1)',
          backgroundColor: 'rgba(15, 118, 110, 0.1)',
          borderWidth: 3,
          fill: true,
          pointRadius: 5,
          pointBackgroundColor: 'rgba(15, 118, 110, 1)',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { font: { size: 11 } } },
          x: { ticks: { font: { size: 11 } }, grid: { display: false } }
        }
      }
    });

    // All Crime Types Bar Chart
    const allTypesCtx = document.getElementById('allTypesChart').getContext('2d');
    const allTypesData = summary.all_types.slice(0, 20); // Show top 20 for readability
    new Chart(allTypesCtx, {
      type: 'bar',
      data: {
        labels: allTypesData.map(t => t.name),
        datasets: [{
          label: 'Total Crimes',
          data: allTypesData.map(t => t.value),
          backgroundColor: allTypesData.map((_, i) => {
            const colors = ['rgba(15, 118, 110, 0.8)', 'rgba(21, 88, 155, 0.8)', 'rgba(217, 119, 6, 0.8)', 'rgba(239, 68, 68, 0.8)', 'rgba(16, 185, 129, 0.8)', 'rgba(168, 85, 247, 0.8)'];
            return colors[i % colors.length];
          }),
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { font: { size: 11 } } },
          y: { ticks: { font: { size: 11 } }, grid: { display: false } }
        }
      }
    });

    // Render Lists with Search
    function renderLists() {
      const getMaxValue = (arr) => Math.max(...arr.map(item => item.value), 1);
      let filteredStates = [...summary.all_states];
      let filteredDistricts = [...summary.all_districts];

      const renderStatesList = () => {
        document.getElementById('all-states-list').innerHTML = filteredStates
          .map(item => {
            const percent = Math.round((item.value / getMaxValue(summary.all_states)) * 100);
            return `<div class="list-item">
              <span class="item-name">${item.name}</span>
              <span class="item-value">${item.value.toLocaleString()}</span>
              <div style="width: 40px; margin-left: 8px;"><div class="progress-bar"><div class="progress-fill" style="width: ${percent}%"></div></div></div>
            </div>`;
          })
          .join('');
      };

      const renderDistrictsList = () => {
        document.getElementById('all-districts-list').innerHTML = filteredDistricts
          .map(item => {
            const percent = Math.round((item.value / getMaxValue(summary.all_districts)) * 100);
            return `<div class="list-item">
              <span class="item-name">${item.name}</span>
              <span class="item-value">${item.value.toLocaleString()}</span>
              <div style="width: 40px; margin-left: 8px;"><div class="progress-bar"><div class="progress-fill" style="width: ${percent}%"></div></div></div>
            </div>`;
          })
          .join('');
      };

      // Search handlers
      document.getElementById('statesSearch').addEventListener('keyup', (e) => {
        const query = e.target.value.toLowerCase();
        filteredStates = summary.all_states.filter(item => item.name.toLowerCase().includes(query));
        renderStatesList();
      });

      document.getElementById('districtsSearch').addEventListener('keyup', (e) => {
        const query = e.target.value.toLowerCase();
        filteredDistricts = summary.all_districts.filter(item => item.name.toLowerCase().includes(query));
        renderDistrictsList();
      });

      // Crime Categories
      const maxCategory = getMaxValue(summary.top_categories);
      document.getElementById('categories-list').innerHTML = summary.top_categories
        .map(item => {
          const percent = Math.round((item.value / maxCategory) * 100);
          return `<div class="list-item">
            <span class="item-name">${item.name}</span>
            <span class="item-value">${item.value.toLocaleString()}</span>
            <div style="width: 40px; margin-left: 8px;"><div class="progress-bar"><div class="progress-fill" style="width: ${percent}%"></div></div></div>
          </div>`;
        })
        .join('');

      // Overview
      document.getElementById('overview-list').innerHTML = `
        <div class="list-item"><span class="item-name">States Covered</span><span class="item-value">${summary.states_count}</span></div>
        <div class="list-item"><span class="item-name">Districts Covered</span><span class="item-value">${summary.districts_count}</span></div>
        <div class="list-item"><span class="item-name">Crime Types</span><span class="item-value">${summary.types_count}</span></div>
        <div class="list-item"><span class="item-name">Year Range</span><span class="item-value">${summary.yearly[0]?.year} - ${summary.yearly[summary.yearly.length - 1]?.year}</span></div>
      `;

      // Years list
      document.getElementById('years-list').innerHTML = summary.yearly
        .map(item => `<div class="list-item"><span class="item-name">${item.year}</span><span class="item-value">${item.value.toLocaleString()}</span></div>`)
        .join('');

      // Initial render
      renderStatesList();
      renderDistrictsList();
    }

    renderStats();
    renderLists();

    // Tab Switching Function
    function switchTab(tabName, btn) {
      // Hide all tab panels
      document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
      });
      
      // Remove active class from all buttons
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
      });
      
      // Show selected tab
      document.getElementById(tabName + '-tab').classList.add('active');
      
      // Add active class to clicked button
      if (btn) {
        btn.classList.add('active');
      }
    }

    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('sw.js').then(reg => {
          reg.update();
        }).catch(() => {});
      });
    }

    let deferredInstallPrompt = null;
    const installBtn = document.getElementById('install-app-btn');
    const installToast = document.getElementById('install-toast');

    function showInstallToast() {
      if (!installToast) return;
      installToast.classList.add('show');
      setTimeout(() => installToast.classList.remove('show'), 2200);
    }

    window.addEventListener('beforeinstallprompt', (event) => {
      event.preventDefault();
      deferredInstallPrompt = event;
      if (installBtn) installBtn.style.display = 'inline-flex';
    });

    window.addEventListener('appinstalled', () => {
      deferredInstallPrompt = null;
      if (installBtn) installBtn.style.display = 'none';
      showInstallToast();
    });

    if (installBtn) {
      installBtn.addEventListener('click', async () => {
        if (deferredInstallPrompt) {
          deferredInstallPrompt.prompt();
          await deferredInstallPrompt.userChoice;
          deferredInstallPrompt = null;
          installBtn.style.display = 'none';
          return;
        }
        alert('Use your browser menu and tap "Add to Home Screen" to install this app.');
      });
    }
  </script>
</body>
</html>
"""

    html_content = html_content.replace("__SUMMARY__", json.dumps(summary))
    output_file.write_text(html_content, encoding="utf-8")


def write_pwa_assets(base_dir: Path) -> None:
    manifest = {
        "name": "Malaysia Crime Heatmap",
        "short_name": "CrimeMap MY",
        "start_url": "index.html",
        "display": "standalone",
        "background_color": "#f8fbff",
        "theme_color": "#0f766e",
        "description": "Interactive Malaysia crime dashboard and heatmap.",
        "icons": [
            {
                "src": "icon.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any"
            }
        ]
    }

    cache_version = f"crime-map-v{int(time.time())}"

    sw_js = """const CACHE_NAME = '__CACHE_NAME__';
const PRECACHE_URLS = [
  './',
  'index.html',
  'crime_heatmap.html',
  'crime_heatmap_preview.html',
  'manifest.webmanifest',
  'icon.svg'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;

  const isHtmlRequest = event.request.mode === 'navigate' ||
    url.pathname.endsWith('.html') ||
    url.pathname === '/' ||
    url.pathname.endsWith('/');

  if (isHtmlRequest) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          if (response && response.status === 200 && response.type === 'basic') {
            const responseClone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, responseClone));
          }
          return response;
        })
        .catch(() => caches.match(event.request).then(cached => cached || caches.match('index.html')))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cached => {
      const networkFetch = fetch(event.request)
        .then(response => {
          if (response && response.status === 200 && response.type === 'basic') {
            const responseClone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, responseClone));
          }
          return response;
        })
        .catch(() => cached);

      return cached || networkFetch;
    })
  );
});
"""
    sw_js = sw_js.replace("__CACHE_NAME__", cache_version)

    icon_svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'>
  <defs>
    <linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>
      <stop offset='0%' stop-color='#0f766e'/>
      <stop offset='100%' stop-color='#15589b'/>
    </linearGradient>
  </defs>
  <rect width='256' height='256' rx='48' fill='url(#g)'/>
  <path d='M128 44c-36 0-66 30-66 66 0 47 66 102 66 102s66-55 66-102c0-36-30-66-66-66zm0 90a24 24 0 1 1 0-48 24 24 0 0 1 0 48z' fill='#fff'/>
</svg>
"""

    (base_dir / "manifest.webmanifest").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (base_dir / "sw.js").write_text(sw_js, encoding="utf-8")
    (base_dir / "icon.svg").write_text(icon_svg, encoding="utf-8")



def main() -> None:
    df = load_data()
    coords_df = load_coordinates()
    heat_data = build_heatmap_from_coordinates(df, coords_df)
    write_html_map(heat_data, OUTPUT_FILE)
    write_preview_map(heat_data, PREVIEW_FILE)
    write_main_page(heat_data, MAIN_PAGE_FILE)
    write_pwa_assets(BASE_DIR)
    print(f"✅ Heatmap generated successfully: {OUTPUT_FILE}")
    print(f"✅ Heatmap preview generated successfully: {PREVIEW_FILE}")
    print(f"✅ Dashboard generated successfully: {MAIN_PAGE_FILE}")
    print(f"✅ PWA assets generated: {BASE_DIR / 'manifest.webmanifest'}, {BASE_DIR / 'sw.js'}")


if __name__ == "__main__":
    main()
