from typing import List, Tuple, Dict
import numpy as np


def assign_risk(score: float) -> str:
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    return "High"


class RiskZoneAnalyzer:
    """Generate simple polygon grid with synthetic risk levels."""

    def __init__(self):
        pass

    def generate_risk_zones(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: float = 0.1,
        threshold_low: float = 0.33,
        threshold_high: float = 0.66,
    ) -> List[Dict[str, Dict]]:
        min_lat, min_lon, max_lat, max_lon = bounds
        lats = np.arange(min_lat, max_lat, resolution)
        lons = np.arange(min_lon, max_lon, resolution)

        zones: List[Dict[str, Dict]] = []
        for lat in lats:
            for lon in lons:
                center_lat = lat + resolution / 2
                center_lon = lon + resolution / 2
                dist = np.sqrt((center_lat - (min_lat + max_lat)/2) ** 2 + (center_lon - (min_lon + max_lon)/2) ** 2)
                score = float(np.exp(-dist * 3))
                score = max(0.0, min(1.0, score))

                if score < threshold_low:
                    risk_level = "low"
                elif score < threshold_high:
                    risk_level = "medium"
                else:
                    risk_level = "high"

                polygon = [
                    [lon, lat],
                    [lon + resolution, lat],
                    [lon + resolution, lat + resolution],
                    [lon, lat + resolution],
                    [lon, lat],
                ]

                zones.append({
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon],
                    },
                    "properties": {
                        "risk_level": risk_level,
                        "risk_score": score,
                    },
                })

        return zones
