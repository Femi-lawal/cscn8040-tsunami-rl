"use client";

import { useEffect, useRef, useMemo } from "react";
import maplibregl from "maplibre-gl";
import type { EpisodeFrame, EventMetadata } from "@/lib/types";
import { ALERT_COLORS } from "@/lib/types";

interface SimulationMapProps {
    frame: EpisodeFrame | null;
    metadata: EventMetadata | null;
    allFrames: EpisodeFrame[];
}

export default function SimulationMap({ frame, metadata, allFrames }: SimulationMapProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const mapRef = useRef<maplibregl.Map | null>(null);
    const epicenterMarkerRef = useRef<maplibregl.Marker | null>(null);
    const sensorMarkersRef = useRef<maplibregl.Marker[]>([]);
    const waveSourceRef = useRef(false);
    const alertBannerRef = useRef<HTMLDivElement | null>(null);

    const epicenter = useMemo(() => {
        if (metadata?.latitude != null && metadata?.longitude != null) {
            return { lat: metadata.latitude, lon: metadata.longitude };
        }
        if (frame?.epicenter) return frame.epicenter;
        return { lat: -2.0, lon: 120.0 };
    }, [metadata, frame]);

    // Initialize map
    useEffect(() => {
        if (!containerRef.current || mapRef.current) return;

        const map = new maplibregl.Map({
            container: containerRef.current,
            style: {
                version: 8,
                sources: {
                    "osm-tiles": {
                        type: "raster",
                        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                        tileSize: 256,
                        attribution: "© OpenStreetMap contributors",
                    },
                },
                layers: [
                    {
                        id: "osm-tiles",
                        type: "raster",
                        source: "osm-tiles",
                        minzoom: 0,
                        maxzoom: 19,
                        paint: {
                            "raster-saturation": -0.8,
                            "raster-brightness-max": 0.4,
                            "raster-contrast": 0.3,
                        },
                    },
                ],
                glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
            },
            center: [epicenter.lon, epicenter.lat],
            zoom: 5,
            attributionControl: false,
        });

        map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

        map.on("load", () => {
            // Wave circle source
            map.addSource("wave-circle", {
                type: "geojson",
                data: { type: "FeatureCollection", features: [] },
            });
            map.addLayer({
                id: "wave-fill",
                type: "fill",
                source: "wave-circle",
                paint: {
                    "fill-color": "#06b6d4",
                    "fill-opacity": 0.08,
                },
            });
            map.addLayer({
                id: "wave-outline",
                type: "line",
                source: "wave-circle",
                paint: {
                    "line-color": "#06b6d4",
                    "line-width": 2,
                    "line-opacity": 0.4,
                },
            });

            // Epicenter pulse source
            map.addSource("epicenter-pulse", {
                type: "geojson",
                data: { type: "FeatureCollection", features: [] },
            });
            map.addLayer({
                id: "epicenter-pulse-layer",
                type: "circle",
                source: "epicenter-pulse",
                paint: {
                    "circle-radius": 20,
                    "circle-color": "#ef4444",
                    "circle-opacity": 0.3,
                    "circle-stroke-width": 2,
                    "circle-stroke-color": "#ef4444",
                    "circle-stroke-opacity": 0.6,
                },
            });

            waveSourceRef.current = true;
        });

        mapRef.current = map;

        return () => {
            map.remove();
            mapRef.current = null;
            waveSourceRef.current = false;
        };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // Recenter when metadata changes
    useEffect(() => {
        if (!mapRef.current) return;
        mapRef.current.flyTo({
            center: [epicenter.lon, epicenter.lat],
            zoom: 5,
            duration: 1000,
        });
    }, [epicenter]);

    // Update epicenter marker
    useEffect(() => {
        if (!mapRef.current) return;

        if (epicenterMarkerRef.current) {
            epicenterMarkerRef.current.remove();
        }

        if (!frame) return;

        const el = document.createElement("div");
        el.className = "epicenter-marker";
        el.innerHTML = `
      <div style="
        width: 20px; height: 20px;
        background: radial-gradient(circle, #ef4444 30%, transparent 70%);
        border-radius: 50%;
        animation: pulse-ring 1.5s ease-out infinite;
        position: relative;
      ">
        <div style="
          position: absolute; inset: 4px;
          background: #ef4444;
          border-radius: 50%;
          border: 2px solid #fca5a5;
        "></div>
      </div>
    `;

        const marker = new maplibregl.Marker({ element: el })
            .setLngLat([epicenter.lon, epicenter.lat])
            .addTo(mapRef.current);

        epicenterMarkerRef.current = marker;
    }, [frame, epicenter]);

    // Update sensors
    useEffect(() => {
        if (!mapRef.current || !frame) return;

        // Clear old
        sensorMarkersRef.current.forEach((m) => m.remove());
        sensorMarkersRef.current = [];

        frame.sensors.forEach((sensor) => {
            if (sensor.lat == null || sensor.lon == null) return;

            const statusColor =
                sensor.status === "triggered" ? "#22c55e" :
                    sensor.status === "monitoring" ? "#3b82f6" : "#4b5563";

            const el = document.createElement("div");
            el.innerHTML = `
        <div style="
          width: 14px; height: 14px;
          background: ${statusColor};
          border-radius: ${sensor.type === "buoy" ? "50%" : "3px"};
          border: 2px solid ${statusColor}88;
          box-shadow: 0 0 ${sensor.status === "triggered" ? "8px" : "0px"} ${statusColor};
          transition: all 0.3s;
        "></div>
      `;
            el.title = `${sensor.id} (${sensor.type}) — ${sensor.status}`;

            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([sensor.lon, sensor.lat])
                .addTo(mapRef.current!);

            sensorMarkersRef.current.push(marker);
        });
    }, [frame]);

    // Update wave circle
    useEffect(() => {
        if (!mapRef.current || !waveSourceRef.current || !frame) return;

        const map = mapRef.current;
        const source = map.getSource("wave-circle") as maplibregl.GeoJSONSource | undefined;
        const pulseSource = map.getSource("epicenter-pulse") as maplibregl.GeoJSONSource | undefined;

        if (pulseSource) {
            pulseSource.setData({
                type: "FeatureCollection",
                features: [
                    {
                        type: "Feature",
                        geometry: {
                            type: "Point",
                            coordinates: [epicenter.lon, epicenter.lat],
                        },
                        properties: {},
                    },
                ],
            });
        }

        if (source && frame.wave_radius_km > 0) {
            const circle = createGeoCircle(epicenter.lon, epicenter.lat, frame.wave_radius_km);
            source.setData({
                type: "FeatureCollection",
                features: [
                    {
                        type: "Feature",
                        geometry: { type: "Polygon", coordinates: [circle] },
                        properties: {},
                    },
                ],
            });

            // Update wave color based on alert
            const alertColor = ALERT_COLORS[frame.alert_level] || "#06b6d4";
            map.setPaintProperty("wave-fill", "fill-color", alertColor);
            map.setPaintProperty("wave-outline", "line-color", alertColor);
        } else if (source) {
            source.setData({ type: "FeatureCollection", features: [] });
        }
    }, [frame, epicenter]);

    const alertColor = frame ? (ALERT_COLORS[frame.alert_level] || "#6b7280") : "#6b7280";

    return (
        <div className="map-container" data-testid="map-container">
            <div ref={containerRef} className="map-canvas" />

            {/* Alert banner overlay */}
            {frame && (
                <div
                    ref={alertBannerRef}
                    className="alert-banner"
                    data-testid="alert-banner"
                    style={{ background: alertColor }}
                >
                    <span className="alert-text">
                        {frame.alert_level.toUpperCase()}
                    </span>
                    <span className="alert-detail">
                        Step {frame.t + 1}/12 · T+{frame.time_min}min
                    </span>
                </div>
            )}

            {/* Event info overlay */}
            {metadata && (
                <div className="event-info-overlay" data-testid="event-info">
                    <span className="event-id">{metadata.event_group_id}</span>
                    <span className="event-location">{metadata.location_name || "Unknown"}</span>
                </div>
            )}

            {/* Legend */}
            <div className="map-legend">
                <div className="legend-item">
                    <span className="legend-dot" style={{ background: "#22c55e" }} />
                    Triggered
                </div>
                <div className="legend-item">
                    <span className="legend-dot" style={{ background: "#3b82f6" }} />
                    Monitoring
                </div>
                <div className="legend-item">
                    <span className="legend-dot" style={{ background: "#4b5563" }} />
                    Inactive
                </div>
                <div className="legend-item">
                    <span className="legend-dot" style={{ background: "#ef4444", borderRadius: 0, width: 10, height: 10 }} />
                    Epicenter
                </div>
            </div>

            <style jsx>{`
        .map-container {
          position: relative;
          width: 100%;
          height: 100%;
          min-height: 300px;
          background: var(--bg-secondary);
        }
        .map-canvas {
          position: absolute;
          inset: 0;
        }
        .alert-banner {
          position: absolute;
          top: 12px;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 8px 20px;
          border-radius: 8px;
          color: white;
          font-weight: 700;
          z-index: 10;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
          transition: background 0.3s;
        }
        .alert-text {
          font-size: 16px;
          letter-spacing: 2px;
        }
        .alert-detail {
          font-size: 12px;
          font-weight: 400;
          opacity: 0.9;
        }
        .event-info-overlay {
          position: absolute;
          top: 12px;
          left: 12px;
          display: flex;
          flex-direction: column;
          gap: 2px;
          background: rgba(10, 14, 23, 0.85);
          padding: 8px 12px;
          border-radius: 6px;
          z-index: 10;
        }
        .event-id {
          font-size: 11px;
          font-weight: 700;
          color: var(--accent-cyan);
          letter-spacing: 0.5px;
        }
        .event-location {
          font-size: 11px;
          color: var(--text-secondary);
        }
        .map-legend {
          position: absolute;
          bottom: 12px;
          left: 12px;
          display: flex;
          gap: 12px;
          background: rgba(10, 14, 23, 0.85);
          padding: 6px 12px;
          border-radius: 6px;
          z-index: 10;
        }
        .legend-item {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 10px;
          color: var(--text-secondary);
        }
        .legend-dot {
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }
      `}</style>
            <style jsx global>{`
        @keyframes pulse-ring {
          0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
          100% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
        }
      `}</style>
        </div>
    );
}

function createGeoCircle(lng: number, lat: number, radiusKm: number, points = 64): [number, number][] {
    const coords: [number, number][] = [];
    const earthRadiusKm = 6371;
    for (let i = 0; i <= points; i++) {
        const angle = (i / points) * 2 * Math.PI;
        const dLat = (radiusKm / earthRadiusKm) * (180 / Math.PI) * Math.cos(angle);
        const dLng =
            ((radiusKm / earthRadiusKm) * (180 / Math.PI) * Math.sin(angle)) /
            Math.cos((lat * Math.PI) / 180);
        coords.push([lng + dLng, lat + dLat]);
    }
    return coords;
}
