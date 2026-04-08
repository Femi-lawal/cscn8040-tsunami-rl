from __future__ import annotations

from pathlib import Path

try:
    from option_a_tsunami_rl.src.data_pipeline import (
        ProjectPaths,
        build_event_summary,
        ensure_dirs,
        load_or_scrape,
        setup_logger,
    )
    from option_a_tsunami_rl.src.enrichment import build_enriched_event_summary
    from option_a_tsunami_rl.src.experiment import run_experiments
    from option_a_tsunami_rl.src.notebook_builder import build_notebook, execute_notebook
except ModuleNotFoundError:
    from src.data_pipeline import ProjectPaths, build_event_summary, ensure_dirs, load_or_scrape, setup_logger
    from src.enrichment import build_enriched_event_summary
    from src.experiment import run_experiments
    from src.notebook_builder import build_notebook, execute_notebook


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = ProjectPaths(root)
    ensure_dirs(paths)
    logger = setup_logger(paths)

    logger.info("Starting Option A tsunami RL pipeline")
    bulletins, details, event_summary = load_or_scrape(paths, logger)
    if not (paths.data_processed / "bmkg_event_summary.csv").exists():
        event_summary = build_event_summary(bulletins, details, paths, logger)

    enriched_event_summary, enrichment_summary = build_enriched_event_summary(event_summary, paths, logger)
    logger.info(
        "External enrichment summary: NOAA matches=%s, USGS matches=%s, external wave proxies=%s",
        int(enrichment_summary.iloc[0]["noaa_matched_event_count"]),
        int(enrichment_summary.iloc[0]["usgs_matched_event_count"]),
        int(enrichment_summary.iloc[0]["external_wave_proxy_count"]),
    )

    run_experiments(root, enriched_event_summary)
    notebook_path = build_notebook(root)
    execute_notebook(notebook_path)
    logger.info("Notebook executed at %s", notebook_path)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
