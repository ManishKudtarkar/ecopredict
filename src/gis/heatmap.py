from typing import List, Tuple, Dict
import numpy as np


class HeatmapGenerator:
    """Generate simple synthetic heatmap data for the API/dashboard.

    This stub creates a grid of risk scores within a bounding box. It does not
    depend on heavy GIS libraries, keeping the dashboard/API responsive even
    without full raster/vector data. Replace the score generation logic with
    your real model outputs when available.
    """

    def __init__(self):
        pass

    def generate_heatmap(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: float = 0.05,
        output_format: str = "json",
    ) -> List[Dict[str, float]]:
        """Generate heatmap points.

        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            resolution: grid step in degrees
            output_format: kept for API compatibility (currently ignored)

        Returns: list of dicts with latitude, longitude, risk_score, risk_category
        """
        min_lat, min_lon, max_lat, max_lon = bounds

        lat_grid = np.arange(min_lat, max_lat + resolution, resolution)
        lon_grid = np.arange(min_lon, max_lon + resolution, resolution)

        points: List[Dict[str, float]] = []

        # Simple synthetic surface: higher risk toward center of the box
        lat_center = (min_lat + max_lat) / 2
        lon_center = (min_lon + max_lon) / 2

        for lat in lat_grid:
            for lon in lon_grid:
                # Gaussian-like decay from center
                dist = np.sqrt((lat - lat_center) ** 2 + (lon - lon_center) ** 2)
                score = float(np.exp(-dist * 3))  # tunable decay factor
                score = max(0.0, min(1.0, score))

                if score < 0.33:
                    category = "low"
                elif score < 0.66:
                    category = "medium"
                else:
                    category = "high"

                points.append({
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "risk_score": score,
                    "risk_category": category,
                })

        return points


# Backward compatibility helper
def generate_heatmap(geo_df, output_path):
    import folium

    m = folium.Map(location=[19.0, 72.8], zoom_start=6)

    folium.Choropleth(
        geo_data=geo_df,
        data=geo_df,
        columns=["region", "risk_score"],
        key_on="feature.properties.region",
        fill_color="YlOrRd"
    ).add_to(m)

    m.save(output_path)
