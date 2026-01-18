#!/usr/bin/env python3
"""Generate maps and visualizations for EcoPredict"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gis.heatmap import HeatmapGenerator
from gis.risk_zones import RiskZoneAnalyzer
from gis.geo_processor import GeoProcessor
from prediction.predict import EcoPredictionEngine
from utils.logger import get_logger
from utils.helpers import load_config, create_directories

logger = get_logger(__name__)


def generate_risk_heatmap(config: dict, output_dir: str):
    """Generate risk heatmap for the region"""
    logger.info("Generating risk heatmap...")
    
    try:
        # Initialize components
        heatmap_gen = HeatmapGenerator()
        prediction_engine = EcoPredictionEngine()
        
        # Load model if available
        model_path = Path("models/trained/best_model.joblib")
        if model_path.exists():
            prediction_engine.load_model(str(model_path))
        
        # Get region bounds from config
        region_bounds = config.get('region_bounds', [72.6, 15.6, 80.9, 22.0])  # Maharashtra
        resolution = config.get('heatmap_resolution', 0.1)
        
        # Generate heatmap
        heatmap_data = heatmap_gen.generate_heatmap(
            bounds=region_bounds,
            resolution=resolution,
            output_format="geojson"
        )
        
        # Save heatmap
        output_path = Path(output_dir) / "risk_heatmap.geojson"
        heatmap_gen.save_heatmap(heatmap_data, str(output_path))
        
        logger.info(f"Risk heatmap saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate risk heatmap: {e}")
        raise


def generate_risk_zones(config: dict, output_dir: str):
    """Generate risk zone boundaries"""
    logger.info("Generating risk zones...")
    
    try:
        # Initialize analyzer
        risk_analyzer = RiskZoneAnalyzer()
        
        # Get parameters from config
        region_bounds = config.get('region_bounds', [72.6, 15.6, 80.9, 22.0])
        resolution = config.get('risk_zone_resolution', 0.05)
        threshold_low = config.get('risk_thresholds', {}).get('low', 0.3)
        threshold_high = config.get('risk_thresholds', {}).get('medium', 0.6)
        
        # Generate risk zones
        risk_zones = risk_analyzer.generate_risk_zones(
            bounds=region_bounds,
            resolution=resolution,
            threshold_low=threshold_low,
            threshold_high=threshold_high
        )
        
        # Save risk zones
        output_path = Path(output_dir) / "risk_zones.geojson"
        risk_analyzer.save_risk_zones(risk_zones, str(output_path))
        
        logger.info(f"Risk zones saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate risk zones: {e}")
        raise


def generate_species_maps(config: dict, output_dir: str):
    """Generate species distribution maps"""
    logger.info("Generating species distribution maps...")
    
    try:
        # Load species data if available
        species_data_path = Path("data/processed/species_metrics.csv")
        
        if not species_data_path.exists():
            logger.warning("Species data not found, generating sample data")
            # Generate sample species data
            bounds = config.get('region_bounds', [72.6, 15.6, 80.9, 22.0])
            n_points = 1000
            
            data = {
                'latitude': np.random.uniform(bounds[1], bounds[3], n_points),
                'longitude': np.random.uniform(bounds[0], bounds[2], n_points),
                'species_count': np.random.poisson(15, n_points),
                'endemic_species': np.random.poisson(2, n_points),
                'threatened_species': np.random.poisson(1, n_points),
                'species_diversity': np.random.uniform(0, 3, n_points)
            }
            
            species_df = pd.DataFrame(data)
        else:
            species_df = pd.read_csv(species_data_path)
        
        # Initialize GIS processor
        geo_processor = GeoProcessor()
        
        # Convert to GeoDataFrame
        species_gdf = geo_processor.points_to_geodataframe(species_df)
        
        # Create species diversity map
        diversity_map = geo_processor.create_interactive_map(
            species_gdf, 'species_diversity', zoom=7
        )
        
        # Save map
        output_path = Path(output_dir) / "species_diversity_map.html"
        diversity_map.save(str(output_path))
        
        logger.info(f"Species diversity map saved to {output_path}")
        
        # Create endemic species map
        if 'endemic_species' in species_gdf.columns:
            endemic_map = geo_processor.create_interactive_map(
                species_gdf, 'endemic_species', zoom=7
            )
            
            output_path = Path(output_dir) / "endemic_species_map.html"
            endemic_map.save(str(output_path))
            
            logger.info(f"Endemic species map saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate species maps: {e}")
        raise


def generate_climate_maps(config: dict, output_dir: str):
    """Generate climate variable maps"""
    logger.info("Generating climate maps...")
    
    try:
        # Load climate data if available
        climate_data_path = Path("data/processed/climate_data.csv")
        
        if not climate_data_path.exists():
            logger.warning("Climate data not found, generating sample data")
            # Generate sample climate data
            bounds = config.get('region_bounds', [72.6, 15.6, 80.9, 22.0])
            n_points = 500
            
            data = {
                'latitude': np.random.uniform(bounds[1], bounds[3], n_points),
                'longitude': np.random.uniform(bounds[0], bounds[2], n_points),
                'temperature': np.random.normal(25, 5, n_points),
                'precipitation': np.random.exponential(2, n_points),
                'humidity': np.random.normal(60, 15, n_points)
            }
            
            climate_df = pd.DataFrame(data)
        else:
            climate_df = pd.read_csv(climate_data_path)
        
        # Initialize GIS processor
        geo_processor = GeoProcessor()
        
        # Convert to GeoDataFrame
        climate_gdf = geo_processor.points_to_geodataframe(climate_df)
        
        # Generate maps for each climate variable
        climate_vars = ['temperature', 'precipitation', 'humidity']
        
        for var in climate_vars:
            if var in climate_gdf.columns:
                climate_map = geo_processor.create_interactive_map(
                    climate_gdf, var, zoom=7
                )
                
                output_path = Path(output_dir) / f"{var}_map.html"
                climate_map.save(str(output_path))
                
                logger.info(f"{var.title()} map saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate climate maps: {e}")
        raise


def generate_all_maps(config_path: str = "config/config.yaml", 
                     output_dir: str = "outputs/maps/"):
    """Generate all maps and visualizations"""
    logger.info("Starting map generation process...")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Create output directory
        create_directories([output_dir])
        
        # Generate different types of maps
        generate_risk_heatmap(config, output_dir)
        generate_risk_zones(config, output_dir)
        generate_species_maps(config, output_dir)
        generate_climate_maps(config, output_dir)
        
        logger.info("All maps generated successfully!")
        
        # Generate summary report
        generate_map_summary(output_dir)
        
    except Exception as e:
        logger.error(f"Map generation failed: {e}")
        raise


def generate_map_summary(output_dir: str):
    """Generate summary of created maps"""
    output_path = Path(output_dir)
    
    # List all generated files
    map_files = list(output_path.glob("*.html")) + list(output_path.glob("*.geojson"))
    
    summary = {
        'generation_timestamp': pd.Timestamp.now().isoformat(),
        'output_directory': str(output_path),
        'total_files': len(map_files),
        'files': [
            {
                'name': f.name,
                'type': f.suffix,
                'size_mb': f.stat().st_size / (1024 * 1024),
                'path': str(f)
            }
            for f in map_files
        ]
    }
    
    # Save summary
    import json
    summary_path = output_path / "map_generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Map generation summary saved to {summary_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate maps for EcoPredict")
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/maps/",
        help="Output directory for maps"
    )
    parser.add_argument(
        "--type",
        choices=["all", "risk", "species", "climate"],
        default="all",
        help="Type of maps to generate"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create output directory
        create_directories([args.output_dir])
        
        # Generate maps based on type
        if args.type == "all":
            generate_all_maps(args.config, args.output_dir)
        elif args.type == "risk":
            generate_risk_heatmap(config, args.output_dir)
            generate_risk_zones(config, args.output_dir)
        elif args.type == "species":
            generate_species_maps(config, args.output_dir)
        elif args.type == "climate":
            generate_climate_maps(config, args.output_dir)
        
        print(f"Maps generated successfully in {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()