from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data_pipeline import ProjectPaths, fetch_text

NOAA_ARCGIS_QUERY = "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/hazards/MapServer/0/query"
USGS_FDSN_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson"
INDONESIA_BBOX = {"min_lon": 90.0, "min_lat": -15.0, "max_lon": 145.0, "max_lat": 15.0}


def _build_url(base_url: str, params: dict[str, object]) -> str:
    request = requests.Request("GET", base_url, params=params).prepare()
    return str(request.url)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def _origin_time_from_parts(row: pd.Series) -> str | None:
    year = int(row["YEAR"])
    month = int(row["MONTH"]) if pd.notna(row["MONTH"]) else 1
    day = int(row["DAY"]) if pd.notna(row["DAY"]) else 1
    hour = int(row["HOUR"]) if pd.notna(row["HOUR"]) else 0
    minute = int(row["MINUTE"]) if pd.notna(row["MINUTE"]) else 0
    second = float(row["SECOND"]) if pd.notna(row["SECOND"]) else 0.0
    second_floor = int(second)
    microsecond = int(round((second - second_floor) * 1_000_000))

    try:
        timestamp = pd.Timestamp(
            year=year,
            month=max(1, month),
            day=max(1, day),
            hour=max(0, hour),
            minute=max(0, minute),
            second=max(0, second_floor),
            microsecond=max(0, microsecond),
            tz="UTC",
        )
    except ValueError:
        return None
    return timestamp.isoformat()


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def fetch_noaa_tsunami_events(
    event_summary: pd.DataFrame,
    paths: ProjectPaths,
    logger: logging.Logger,
) -> pd.DataFrame:
    cache_path = paths.data_processed / "noaa_tsunami_events.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "noaa_event_id" in cached.columns and "event_id" not in cached.columns:
            cached = cached.rename(columns={"noaa_event_id": "event_id"})
        return cached

    session = requests.Session()
    records: list[dict] = []
    offset = 0
    page = 1

    while True:
        params = {
            "where": "YEAR >= 2000",
            "geometry": f"{INDONESIA_BBOX['min_lon']},{INDONESIA_BBOX['min_lat']},{INDONESIA_BBOX['max_lon']},{INDONESIA_BBOX['max_lat']}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(
                [
                    "ID",
                    "YEAR",
                    "MONTH",
                    "DAY",
                    "HOUR",
                    "MINUTE",
                    "SECOND",
                    "LATITUDE",
                    "LONGITUDE",
                    "LOCATION_NAME",
                    "COUNTRY",
                    "REGION",
                    "CAUSE",
                    "EVENT_VALIDITY",
                    "EQ_MAGNITUDE",
                    "EQ_DEPTH",
                    "MAX_EVENT_RUNUP",
                    "TS_INTENSITY",
                    "DEATHS_TOTAL",
                    "NUM_RUNUP",
                    "URL",
                ]
            ),
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": 2000,
            "f": "json",
        }
        url = _build_url(NOAA_ARCGIS_QUERY, params)
        raw_text = fetch_text(session, url, paths.search_log, timeout=60)
        (paths.raw_noaa / f"tsunami_events_page_{page}.json").write_text(raw_text, encoding="utf-8")
        payload = json.loads(raw_text)
        features = payload.get("features", [])
        if not features:
            break

        for feature in features:
            attr = feature.get("attributes", {})
            records.append(
                {
                    "event_id": attr.get("ID"),
                    "origin_time_utc": _origin_time_from_parts(pd.Series(attr)),
                    "latitude": attr.get("LATITUDE"),
                    "longitude": attr.get("LONGITUDE"),
                    "location_name": attr.get("LOCATION_NAME"),
                    "country": attr.get("COUNTRY"),
                    "region": attr.get("REGION"),
                    "cause": attr.get("CAUSE"),
                    "event_validity": attr.get("EVENT_VALIDITY"),
                    "eq_magnitude": attr.get("EQ_MAGNITUDE"),
                    "eq_depth_km": attr.get("EQ_DEPTH"),
                    "max_event_runup_m": attr.get("MAX_EVENT_RUNUP"),
                    "tsunami_intensity": attr.get("TS_INTENSITY"),
                    "deaths_total": attr.get("DEATHS_TOTAL"),
                    "num_runup": attr.get("NUM_RUNUP"),
                    "url": attr.get("URL"),
                }
            )

        if len(features) < 2000:
            break
        offset += len(features)
        page += 1

    noaa_df = pd.DataFrame(records)
    noaa_df.to_csv(cache_path, index=False)
    logger.info("Saved %s NOAA tsunami events", len(noaa_df))
    return noaa_df


def fetch_usgs_catalog_events(
    event_summary: pd.DataFrame,
    paths: ProjectPaths,
    logger: logging.Logger,
) -> pd.DataFrame:
    cache_path = paths.data_processed / "usgs_earthquake_events.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "usgs_event_id" in cached.columns and "event_id" not in cached.columns:
            cached = cached.rename(columns={"usgs_event_id": "event_id"})
        return cached

    origin_times = _normalize_timestamp(event_summary["origin_time_utc"])
    start_time = (origin_times.min() - pd.Timedelta(days=2)).date().isoformat()
    end_time = (origin_times.max() + pd.Timedelta(days=2)).date().isoformat()

    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": 6.0,
        "minlatitude": INDONESIA_BBOX["min_lat"],
        "maxlatitude": INDONESIA_BBOX["max_lat"],
        "minlongitude": INDONESIA_BBOX["min_lon"],
        "maxlongitude": INDONESIA_BBOX["max_lon"],
        "orderby": "time-asc",
        "limit": 20000,
    }
    session = requests.Session()
    url = _build_url(USGS_FDSN_QUERY, params)
    raw_text = fetch_text(session, url, paths.search_log, timeout=60)
    (paths.raw_usgs / "earthquake_catalog.geojson").write_text(raw_text, encoding="utf-8")
    payload = json.loads(raw_text)

    records: list[dict] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [None, None, None])
        event_time = properties.get("time")
        origin_time = pd.to_datetime(event_time, unit="ms", utc=True, errors="coerce")
        records.append(
            {
                "event_id": feature.get("id"),
                "origin_time_utc": origin_time.isoformat() if pd.notna(origin_time) else None,
                "longitude": coordinates[0],
                "latitude": coordinates[1],
                "depth_km": coordinates[2],
                "magnitude": properties.get("mag"),
                "place": properties.get("place"),
                "tsunami_flag": properties.get("tsunami"),
                "significance": properties.get("sig"),
                "mmi": properties.get("mmi"),
                "felt": properties.get("felt"),
                "status": properties.get("status"),
                "url": properties.get("url"),
            }
        )

    usgs_df = pd.DataFrame(records)
    usgs_df.to_csv(cache_path, index=False)
    logger.info("Saved %s USGS earthquake events", len(usgs_df))
    return usgs_df


def _match_catalog(
    bmkg_events: pd.DataFrame,
    external_df: pd.DataFrame,
    *,
    external_time_col: str,
    external_lat_col: str,
    external_lon_col: str,
    max_time_minutes: float,
    max_distance_km: float,
    prefix: str,
    external_columns: list[str],
) -> pd.DataFrame:
    if external_df.empty:
        return pd.DataFrame({"event_group_id": bmkg_events["event_group_id"]})

    bmkg = bmkg_events[["event_group_id", "origin_time_utc", "latitude", "longitude"]].copy()
    bmkg["origin_time_utc"] = _normalize_timestamp(bmkg["origin_time_utc"])
    external = external_df.copy()
    external[external_time_col] = _normalize_timestamp(external[external_time_col])
    external = external.dropna(subset=[external_time_col, external_lat_col, external_lon_col]).copy()

    min_origin = bmkg["origin_time_utc"].min() - pd.Timedelta(days=30)
    max_origin = bmkg["origin_time_utc"].max() + pd.Timedelta(days=30)
    external = external[
        (external[external_time_col] >= min_origin) & (external[external_time_col] <= max_origin)
    ].copy()

    matched_rows: list[dict] = []
    time_window = pd.Timedelta(minutes=max_time_minutes)

    for row in bmkg.itertuples(index=False):
        if pd.isna(row.origin_time_utc) or pd.isna(row.latitude) or pd.isna(row.longitude):
            matched_rows.append({"event_group_id": row.event_group_id})
            continue

        candidates = external[
            external[external_time_col].sub(row.origin_time_utc).abs() <= time_window
        ].copy()
        candidates = candidates.dropna(subset=[external_lat_col, external_lon_col])
        if candidates.empty:
            matched_rows.append({"event_group_id": row.event_group_id})
            continue

        distances = _haversine_km(
            float(row.latitude),
            float(row.longitude),
            candidates[external_lat_col].to_numpy(dtype=float),
            candidates[external_lon_col].to_numpy(dtype=float),
        )
        candidates[f"{prefix}_match_km"] = distances
        candidates[f"{prefix}_match_minutes"] = (
            candidates[external_time_col].sub(row.origin_time_utc).abs().dt.total_seconds() / 60.0
        )
        candidates = candidates[candidates[f"{prefix}_match_km"] <= max_distance_km]
        if candidates.empty:
            matched_rows.append({"event_group_id": row.event_group_id})
            continue

        candidates[f"{prefix}_score"] = (
            candidates[f"{prefix}_match_minutes"] + 0.25 * candidates[f"{prefix}_match_km"]
        )
        best = candidates.sort_values(f"{prefix}_score").iloc[0]
        matched = {"event_group_id": row.event_group_id}
        for column in external_columns:
            matched[f"{prefix}_{column}"] = best.get(column)
        matched[f"{prefix}_match_minutes"] = best.get(f"{prefix}_match_minutes")
        matched[f"{prefix}_match_km"] = best.get(f"{prefix}_match_km")
        matched_rows.append(matched)

    return pd.DataFrame(matched_rows)


def build_enriched_event_summary(
    event_summary: pd.DataFrame,
    paths: ProjectPaths,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched_path = paths.data_processed / "bmkg_event_summary_enriched.csv"
    summary_path = paths.data_processed / "external_enrichment_summary.csv"
    if enriched_path.exists() and summary_path.exists():
        return pd.read_csv(enriched_path), pd.read_csv(summary_path)

    noaa_df = fetch_noaa_tsunami_events(event_summary, paths, logger)
    usgs_df = fetch_usgs_catalog_events(event_summary, paths, logger)

    noaa_matches = _match_catalog(
        event_summary,
        noaa_df,
        external_time_col="origin_time_utc",
        external_lat_col="latitude",
        external_lon_col="longitude",
        max_time_minutes=240.0,
        max_distance_km=350.0,
        prefix="noaa",
        external_columns=[
            "event_id",
            "eq_magnitude",
            "eq_depth_km",
            "max_event_runup_m",
            "tsunami_intensity",
            "deaths_total",
            "num_runup",
            "event_validity",
            "country",
            "region",
            "cause",
            "url",
        ],
    )
    usgs_matches = _match_catalog(
        event_summary,
        usgs_df,
        external_time_col="origin_time_utc",
        external_lat_col="latitude",
        external_lon_col="longitude",
        max_time_minutes=180.0,
        max_distance_km=250.0,
        prefix="usgs",
        external_columns=[
            "event_id",
            "magnitude",
            "depth_km",
            "tsunami_flag",
            "significance",
            "mmi",
            "felt",
            "status",
            "place",
            "url",
        ],
    )

    enriched = event_summary.merge(noaa_matches, on="event_group_id", how="left")
    enriched = enriched.merge(usgs_matches, on="event_group_id", how="left")
    enriched["data_source"] = "bmkg"
    enriched["is_synthetic"] = 0

    enriched["external_wave_proxy_m"] = pd.to_numeric(enriched["noaa_max_event_runup_m"], errors="coerce")
    enriched["external_deaths_total"] = pd.to_numeric(enriched["noaa_deaths_total"], errors="coerce")
    enriched["wave_height_source"] = np.where(
        enriched["observed_max_wave_m"].notna(),
        enriched["wave_height_source"],
        np.where(enriched["external_wave_proxy_m"].notna(), "noaa_runup_proxy", "imputed"),
    )
    enriched["training_weight"] = np.where(
        enriched["usgs_tsunami_flag"].fillna(0).astype(float) >= 1,
        np.maximum(enriched["training_weight"], 4),
        enriched["training_weight"],
    )

    enrichment_summary = pd.DataFrame(
        [
            {
                "bmkg_event_count": len(event_summary),
                "noaa_catalog_count": len(noaa_df),
                "usgs_catalog_count": len(usgs_df),
                "noaa_matched_event_count": int(enriched["noaa_max_event_runup_m"].notna().sum()),
                "usgs_matched_event_count": int(enriched["usgs_magnitude"].notna().sum()),
                "bmkg_observed_wave_count": int(enriched["observed_max_wave_m"].notna().sum()),
                "external_wave_proxy_count": int(enriched["external_wave_proxy_m"].notna().sum()),
            }
        ]
    )

    enriched.to_csv(enriched_path, index=False)
    enrichment_summary.to_csv(summary_path, index=False)
    logger.info(
        "Saved enriched event summary with %s NOAA matches and %s USGS matches",
        int(enriched["noaa_max_event_runup_m"].notna().sum()),
        int(enriched["usgs_magnitude"].notna().sum()),
    )
    return enriched, enrichment_summary


def generate_synthetic_training_catalog(
    train_catalog: pd.DataFrame,
    *,
    synthetic_multiplier: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_catalog = train_catalog.copy().reset_index(drop=True)
    if train_catalog.empty:
        return train_catalog.iloc[0:0].copy()

    train_catalog["origin_time_utc"] = _normalize_timestamp(train_catalog["origin_time_utc"])
    numeric_cols = [
        "initial_magnitude",
        "max_magnitude",
        "initial_depth_km",
        "final_depth_km",
        "coastal_proximity_index",
        "first_bulletin_delay_min",
        "final_bulletin_delay_min",
        "observed_max_wave_m",
        "external_wave_proxy_m",
        "noaa_max_event_runup_m",
        "noaa_tsunami_intensity",
        "usgs_magnitude",
        "usgs_depth_km",
        "usgs_significance",
        "usgs_mmi",
    ]
    bool_cols = [
        "confirmed_threat_flag",
        "potential_threat_flag",
        "no_threat_flag",
        "sea_level_confirmed_flag",
        "has_threat_assessment",
    ]
    numeric_cols = [column for column in numeric_cols if column in train_catalog.columns]
    bool_cols = [column for column in bool_cols if column in train_catalog.columns]

    synthetic_rows: list[dict] = []
    for danger_label, group in train_catalog.groupby("danger_label", sort=False):
        synth_count = max(1, int(round(len(group) * synthetic_multiplier)))
        numeric_std = group[numeric_cols].apply(pd.to_numeric, errors="coerce").std(ddof=0).fillna(0.0)
        bool_prob = group[bool_cols].fillna(False).mean() if bool_cols else pd.Series(dtype=float)
        min_time = group["origin_time_utc"].min()
        max_time = group["origin_time_utc"].max()

        for index in range(synth_count):
            base = group.iloc[int(rng.integers(len(group)))].copy()
            row = base.to_dict()
            for column in numeric_cols:
                if column not in row or pd.isna(row[column]):
                    continue
                scale = max(float(numeric_std.get(column, 0.0)), 1e-6)
                row[column] = float(row[column]) + float(rng.normal(0.0, 0.15 * scale))

            row["initial_magnitude"] = float(np.clip(row.get("initial_magnitude", 7.0), 6.5, 9.5))
            row["max_magnitude"] = float(max(row["initial_magnitude"], row.get("max_magnitude", row["initial_magnitude"])))
            row["initial_depth_km"] = float(max(1.0, row.get("initial_depth_km", 20.0)))
            row["final_depth_km"] = float(max(1.0, row.get("final_depth_km", row["initial_depth_km"])))
            row["coastal_proximity_index"] = float(np.clip(row.get("coastal_proximity_index", 0.5), 0.0, 1.0))
            row["first_bulletin_delay_min"] = float(max(0.5, row.get("first_bulletin_delay_min", 8.0)))
            row["final_bulletin_delay_min"] = float(max(row["first_bulletin_delay_min"], row.get("final_bulletin_delay_min", row["first_bulletin_delay_min"])))
            row["observed_max_wave_m"] = (
                float(max(0.0, row["observed_max_wave_m"]))
                if pd.notna(row.get("observed_max_wave_m"))
                else row.get("observed_max_wave_m")
            )
            row["external_wave_proxy_m"] = (
                float(max(0.0, row["external_wave_proxy_m"]))
                if pd.notna(row.get("external_wave_proxy_m"))
                else row.get("external_wave_proxy_m")
            )
            row["bulletin_count"] = int(max(1, round(float(row.get("bulletin_count", 1)))))
            row["max_bulletin_number"] = int(max(row["bulletin_count"], round(float(row.get("max_bulletin_number", row["bulletin_count"])))))
            row["training_weight"] = float(max(1.0, 0.75 * float(row.get("training_weight", 1.0))))

            for column in bool_cols:
                row[column] = bool(rng.random() < float(bool_prob.get(column, 0.0)))

            if pd.notna(min_time) and pd.notna(max_time):
                jitter_seconds = float(rng.uniform(0.0, max((max_time - min_time).total_seconds(), 1.0)))
                row["origin_time_utc"] = (min_time + timedelta(seconds=jitter_seconds)).isoformat()
            else:
                row["origin_time_utc"] = pd.Timestamp.utcnow().tz_localize("UTC").isoformat()

            row["event_group_id"] = f"synthetic_{danger_label}_{index:04d}"
            row["data_source"] = "synthetic_bootstrap"
            row["is_synthetic"] = 1
            if pd.notna(row.get("observed_max_wave_m")):
                row["wave_height_source"] = "synthetic_bootstrap"

            synthetic_rows.append(row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    synthetic_df = synthetic_df[train_catalog.columns.tolist()]
    return synthetic_df.reset_index(drop=True)
