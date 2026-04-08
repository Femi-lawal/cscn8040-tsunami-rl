from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_BULLETIN_LIST = "https://rtsp.bmkg.go.id/publicbull.php"
BASE_DETAIL = "https://rtsp.bmkg.go.id/publicdetail.php"
EXPECTED_HEADERS = [
    "Date",
    "Time (UTC)",
    "Magnitude",
    "Depth (Km)",
    "Latitude",
    "Longitude",
    "Location",
    "Type",
    "Bulletin Number",
    "Bulletin Type",
    "Event Group",
]
USER_AGENT = "Mozilla/5.0 (compatible; option-a-coursework-bot/1.0)"
SEARCH_LOG_HEADERS = [
    "timestamp_utc",
    "url",
    "http_status",
    "bytes",
    "elapsed_ms",
    "error_message",
]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def raw_publicbull(self) -> Path:
        return self.data_raw / "publicbull_pages"

    @property
    def raw_detail(self) -> Path:
        return self.data_raw / "detail_pages"

    @property
    def raw_external(self) -> Path:
        return self.data_raw / "external"

    @property
    def raw_noaa(self) -> Path:
        return self.raw_external / "noaa"

    @property
    def raw_usgs(self) -> Path:
        return self.raw_external / "usgs"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def search_log(self) -> Path:
        return self.logs / "search_log.csv"

    @property
    def pipeline_log(self) -> Path:
        return self.logs / "pipeline.log"


def ensure_dirs(paths: ProjectPaths) -> None:
    for directory in [
        paths.raw_publicbull,
        paths.raw_detail,
        paths.raw_noaa,
        paths.raw_usgs,
        paths.data_processed,
        paths.logs,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    if not paths.search_log.exists():
        with paths.search_log.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(SEARCH_LOG_HEADERS)


def setup_logger(paths: ProjectPaths) -> logging.Logger:
    logger = logging.getLogger("option_a_tsunami_rl")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(paths.pipeline_log, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log_fetch(
    search_log_path: Path,
    url: str,
    status: int,
    content_bytes: int,
    elapsed_ms: float,
    error_message: str = "",
) -> None:
    with search_log_path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                url,
                status,
                content_bytes,
                round(elapsed_ms, 1),
                error_message,
            ]
        )


def fetch_text(
    session: requests.Session,
    url: str,
    search_log_path: Path,
    *,
    timeout: int = 30,
) -> str:
    started = time.monotonic()
    response = session.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    log_fetch(
        search_log_path,
        url,
        response.status_code,
        len(response.content),
        elapsed_ms,
        "" if response.ok else response.reason,
    )
    response.raise_for_status()
    return response.text


def _normalize_cell_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _parse_numeric(value: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    if not match:
        return None
    return float(match.group(1))


def _parse_lat_lon(value: str) -> float | None:
    if not value:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([NSEW])", value.strip(), re.I)
    if not match:
        return _parse_numeric(value)
    number = float(match.group(1))
    direction = match.group(2).upper()
    if direction in {"S", "W"}:
        number *= -1
    return number


def _parse_origin_time(date_text: str, time_text: str) -> str | None:
    try:
        dt = datetime.strptime(
            f"{date_text} {time_text}",
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _parse_issue_time(detail_event_id: str | None) -> str | None:
    if not detail_event_id or not re.fullmatch(r"\d{14}", detail_event_id):
        return None
    dt = datetime.strptime(detail_event_id, "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )
    return dt.isoformat()


def _extract_total_pages(html: str) -> int:
    text = " ".join(BeautifulSoup(html, "html.parser").get_text(" ", strip=True).split())
    match = re.search(r"Page(?:\s+\d+)?\s+of\s+(\d+)\s+Event", text)
    if match:
        return int(match.group(1))
    match = re.search(r"Page(?:\s+\d+)?\s+of\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    return 1


def parse_publicbull_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows: list[dict] = []
        header_seen = False
        for tr in table.find_all("tr", recursive=False):
            tds = tr.find_all("td", recursive=False)
            if len(tds) != len(EXPECTED_HEADERS):
                continue

            texts = [_normalize_cell_text(td.get_text(" ", strip=True)) for td in tds]
            if texts == EXPECTED_HEADERS:
                header_seen = True
                continue
            if not header_seen:
                continue

            location_link = tds[6].find("a", href=True)
            event_group_link = tds[10].find("a", href=True)
            detail_event_id = None
            if location_link is not None:
                match = re.search(r"eventid=([^&]+)", location_link["href"])
                if match:
                    detail_event_id = match.group(1)

            timeline_group_id = None
            if event_group_link is not None:
                match = re.search(r"grup=([^&]+)", event_group_link["href"])
                if match:
                    timeline_group_id = match.group(1)

            bulletin_number_match = re.search(
                r"(\d+)(?:\s+([0-9.]+))?", texts[8]
            )
            rows.append(
                {
                    "origin_date": texts[0],
                    "origin_clock_utc": texts[1],
                    "origin_time_utc": _parse_origin_time(texts[0], texts[1]),
                    "magnitude": _parse_numeric(texts[2]),
                    "depth_km": _parse_numeric(texts[3]),
                    "latitude_raw": texts[4],
                    "longitude_raw": texts[5],
                    "latitude": _parse_lat_lon(texts[4]),
                    "longitude": _parse_lat_lon(texts[5]),
                    "location_name": texts[6],
                    "event_type": texts[7],
                    "bulletin_number_raw": texts[8],
                    "bulletin_number": (
                        int(bulletin_number_match.group(1))
                        if bulletin_number_match
                        else None
                    ),
                    "bulletin_revision": (
                        bulletin_number_match.group(2)
                        if bulletin_number_match and bulletin_number_match.group(2)
                        else None
                    ),
                    "bulletin_type": texts[9],
                    "event_group_id": texts[10],
                    "timeline_group_id": timeline_group_id or texts[10],
                    "detail_event_id": detail_event_id,
                    "issue_time_utc": _parse_issue_time(detail_event_id),
                    "detail_url": (
                        f"{BASE_DETAIL}?eventid={detail_event_id}"
                        if detail_event_id
                        else None
                    ),
                }
            )
        if rows:
            return rows
    return []


def _clean_detail_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    lines = [
        _normalize_cell_text(line)
        for line in soup.get_text("\n").splitlines()
        if _normalize_cell_text(line)
    ]
    return "\n".join(lines)


def _extract_section(text: str, start_label: str, end_labels: Iterable[str]) -> str | None:
    start = text.find(start_label)
    if start < 0:
        return None
    end = len(text)
    for label in end_labels:
        position = text.find(label, start + len(start_label))
        if position >= 0:
            end = min(end, position)
    return text[start:end].strip()


def parse_detail_page(detail_event_id: str, html: str) -> dict:
    text = _clean_detail_text(html)
    compact_text = " ".join(text.split())

    no_threat_flag = bool(
        re.search(
            r"NO TSUNAMI THREAT|not capable of generating a tsunami",
            compact_text,
            re.I,
        )
    )
    potential_threat_flag = bool(
        re.search(
            r"POTENTIAL TSUNAMI THREAT|evaluating this earthquake to determine if a tsunami has been generated",
            compact_text,
            re.I,
        )
    )
    confirmed_threat_flag = bool(
        re.search(
            r"CONFIRMED TSUNAMI THREAT|TSUNAMI WAS GENERATED",
            compact_text,
            re.I,
        )
    )
    monitoring_flag = bool(
        re.search(r"monitor sea level gauges", compact_text, re.I)
    )
    sea_level_confirmed_flag = bool(
        re.search(r"Sea level observations have confirmed", compact_text, re.I)
    )

    observed_amplitudes: list[float] = []
    capture_amplitudes = False
    for line in text.splitlines():
        if "AMPL(m)" in line or "Maximum wave amplitudes observed" in line:
            capture_amplitudes = True
            continue
        if not capture_amplitudes:
            continue
        if any(stop_token in line for stop_token in ["3. ADVICE", "4. ADVICE", "5. UPDATES", "END OF BULLETIN"]):
            break
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s+\d{2}:\d{2}\b", line)
        if match:
            observed_amplitudes.append(float(match.group(1)))

    if not observed_amplitudes:
        amplitude_matches = re.findall(
            r"\b([0-9]+(?:\.[0-9]+)?)\s+\d{2}:\d{2}\s+[A-Z][a-z]{2}\s+\d{2},\s+\d{4}",
            compact_text,
        )
        observed_amplitudes = [float(value) for value in amplitude_matches]

    bulletin_match = re.search(
        r"Bulletin\s*:\s*(.+?)(?:TSP-InaTEWS|PUBLIC TSUNAMI BULLETIN|TSUNAMI BULLETIN NUMBER)",
        compact_text,
        re.I,
    )
    evaluation_text = _extract_section(
        text,
        "2. EVALUATION",
        ["3. ADVICE", "3. UPDATES", "4. ADVICE", "4. UPDATES", "END OF BULLETIN"],
    )

    if confirmed_threat_flag:
        threat_class = "confirmed"
    elif potential_threat_flag or monitoring_flag:
        threat_class = "potential"
    elif no_threat_flag:
        threat_class = "no_threat"
    else:
        threat_class = "unknown"

    return {
        "detail_event_id": detail_event_id,
        "detail_text": text,
        "bulletin_header": bulletin_match.group(1).strip() if bulletin_match else None,
        "evaluation_text": evaluation_text,
        "no_threat_flag": no_threat_flag,
        "potential_threat_flag": potential_threat_flag,
        "confirmed_threat_flag": confirmed_threat_flag,
        "monitoring_flag": monitoring_flag,
        "sea_level_confirmed_flag": sea_level_confirmed_flag,
        "observed_max_wave_m": max(observed_amplitudes) if observed_amplitudes else None,
        "threat_class": threat_class,
    }


def scrape_bmkg_archive(
    paths: ProjectPaths,
    logger: logging.Logger,
    *,
    page_pause_seconds: float = 0.15,
    detail_pause_seconds: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = requests.Session()

    first_page_url = f"{BASE_BULLETIN_LIST}?halaman=1"
    logger.info("Fetching bulletin index page: %s", first_page_url)
    first_html = fetch_text(session, first_page_url, paths.search_log)
    (paths.raw_publicbull / "page_1.html").write_text(first_html, encoding="utf-8")

    page_count = _extract_total_pages(first_html)
    bulletin_rows = parse_publicbull_page(first_html)
    logger.info("Detected %s bulletin pages", page_count)

    for page in range(2, page_count + 1):
        url = f"{BASE_BULLETIN_LIST}?halaman={page}"
        html = fetch_text(session, url, paths.search_log)
        (paths.raw_publicbull / f"page_{page}.html").write_text(
            html, encoding="utf-8"
        )
        bulletin_rows.extend(parse_publicbull_page(html))
        if page_pause_seconds:
            time.sleep(page_pause_seconds)

    bulletins = pd.DataFrame(bulletin_rows).drop_duplicates(
        subset=["detail_event_id", "event_group_id", "bulletin_number_raw"]
    ).sort_values(
        ["event_group_id", "issue_time_utc", "bulletin_number"],
        na_position="last",
    )
    bulletins.to_csv(paths.data_processed / "bmkg_public_bulletins.csv", index=False)
    logger.info("Saved %s bulletin rows", len(bulletins))

    detail_records: list[dict] = []
    unique_detail_ids = [
        detail_id
        for detail_id in bulletins["detail_event_id"].dropna().astype(str).unique().tolist()
        if detail_id
    ]
    logger.info("Fetching %s detail pages", len(unique_detail_ids))

    for index, detail_event_id in enumerate(unique_detail_ids, start=1):
        detail_url = f"{BASE_DETAIL}?eventid={detail_event_id}"
        html = fetch_text(session, detail_url, paths.search_log)
        (paths.raw_detail / f"{detail_event_id}.html").write_text(
            html, encoding="utf-8"
        )
        detail_records.append(parse_detail_page(detail_event_id, html))
        if detail_pause_seconds:
            time.sleep(detail_pause_seconds)
        if index % 100 == 0 or index == len(unique_detail_ids):
            logger.info("Fetched %s/%s detail pages", index, len(unique_detail_ids))

    details = pd.DataFrame(detail_records)
    details.to_csv(paths.data_processed / "bmkg_detail_pages.csv", index=False)
    logger.info("Saved %s parsed detail records", len(details))
    return bulletins, details


def _minutes_between(start_iso: str | None, end_iso: str | None) -> float | None:
    if not start_iso or not end_iso:
        return None
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except ValueError:
        return None
    return (end - start).total_seconds() / 60.0


def _coastal_proximity_index(location_name: str, latitude: float | None) -> float:
    location_text = str(location_name).lower()
    score = 0.35
    if any(
        token in location_text
        for token in ["near coast", "off west coast", "off east coast", "coast of"]
    ):
        score += 0.35
    if any(token in location_text for token in ["sea", "strait", "bay", "gulf"]):
        score += 0.15
    if "indonesia" in location_text:
        score += 0.10
    if latitude is not None and abs(latitude) < 10:
        score += 0.05
    return min(score, 1.0)


def build_event_summary(
    bulletins: pd.DataFrame,
    details: pd.DataFrame,
    paths: ProjectPaths,
    logger: logging.Logger,
) -> pd.DataFrame:
    merged = bulletins.merge(details, on="detail_event_id", how="left")
    merged["issue_delay_min"] = merged.apply(
        lambda row: _minutes_between(row["origin_time_utc"], row["issue_time_utc"]),
        axis=1,
    )
    merged.to_csv(paths.data_processed / "bmkg_bulletins_enriched.csv", index=False)

    real_events = merged[
        merged["event_type"].astype(str).str.upper().eq("REAL EVENT")
    ].copy()

    summary_rows: list[dict] = []
    for event_group_id, group in real_events.groupby("event_group_id", sort=False):
        group = group.sort_values(["issue_time_utc", "bulletin_number"], na_position="last")
        first_row = group.iloc[0]
        last_row = group.iloc[-1]

        confirmed_flag = bool(group["confirmed_threat_flag"].fillna(False).any())
        potential_flag = bool(group["potential_threat_flag"].fillna(False).any())
        no_threat_flag = bool(group["no_threat_flag"].fillna(False).any())
        has_threat_assessment = bool(
            group["bulletin_type"].astype(str).str.contains("THREAT", case=False, na=False).any()
        )
        sea_level_confirmed_flag = bool(
            group["sea_level_confirmed_flag"].fillna(False).any()
        )
        max_wave = group["observed_max_wave_m"].dropna().max()

        if confirmed_flag or sea_level_confirmed_flag or (pd.notna(max_wave) and max_wave >= 0.08):
            danger_tier = 2
            danger_label = "confirmed_threat"
            target_alert_level = 4
        elif potential_flag or (has_threat_assessment and not no_threat_flag):
            danger_tier = 1
            danger_label = "potential_threat"
            target_alert_level = 2
        else:
            danger_tier = 0
            danger_label = "no_threat"
            target_alert_level = 0

        summary_rows.append(
            {
                "event_group_id": event_group_id,
                "origin_time_utc": first_row["origin_time_utc"],
                "location_name": first_row["location_name"],
                "latitude": first_row["latitude"],
                "longitude": first_row["longitude"],
                "initial_magnitude": first_row["magnitude"],
                "max_magnitude": group["magnitude"].dropna().max(),
                "initial_depth_km": first_row["depth_km"],
                "final_depth_km": last_row["depth_km"],
                "coastal_proximity_index": _coastal_proximity_index(
                    str(first_row["location_name"]),
                    first_row["latitude"],
                ),
                "first_bulletin_delay_min": group["issue_delay_min"].dropna().min(),
                "final_bulletin_delay_min": group["issue_delay_min"].dropna().max(),
                "bulletin_count": int(group["detail_event_id"].nunique()),
                "max_bulletin_number": int(group["bulletin_number"].fillna(0).max()),
                "danger_tier": danger_tier,
                "danger_label": danger_label,
                "target_alert_level": target_alert_level,
                "confirmed_threat_flag": confirmed_flag,
                "potential_threat_flag": potential_flag,
                "no_threat_flag": no_threat_flag,
                "sea_level_confirmed_flag": sea_level_confirmed_flag,
                "observed_max_wave_m": max_wave if pd.notna(max_wave) else None,
                "wave_height_source": "observed" if pd.notna(max_wave) else "imputed",
                "has_threat_assessment": has_threat_assessment,
                "training_weight": 6 if confirmed_flag else (3 if danger_tier == 1 else 1),
            }
        )

    event_summary = pd.DataFrame(summary_rows).sort_values("origin_time_utc")
    event_summary.to_csv(paths.data_processed / "bmkg_event_summary.csv", index=False)

    class_counts = event_summary["danger_label"].value_counts().to_dict()
    logger.info("Built event summary for %s real events", len(event_summary))
    logger.info("Danger-label distribution: %s", class_counts)
    return event_summary


def load_or_scrape(
    paths: ProjectPaths,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bulletins_path = paths.data_processed / "bmkg_public_bulletins.csv"
    details_path = paths.data_processed / "bmkg_detail_pages.csv"
    events_path = paths.data_processed / "bmkg_event_summary.csv"

    if bulletins_path.exists() and details_path.exists() and events_path.exists():
        logger.info("Using cached processed data")
        return (
            pd.read_csv(bulletins_path),
            pd.read_csv(details_path),
            pd.read_csv(events_path),
        )

    if bulletins_path.exists() and details_path.exists() and not events_path.exists():
        logger.info("Rebuilding event summary from cached bulletin and detail tables")
        bulletins = pd.read_csv(bulletins_path)
        details = pd.read_csv(details_path)
        event_summary = build_event_summary(bulletins, details, paths, logger)
        return bulletins, details, event_summary

    bulletins, details = scrape_bmkg_archive(paths, logger)
    event_summary = build_event_summary(bulletins, details, paths, logger)
    return bulletins, details, event_summary
