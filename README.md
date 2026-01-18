# EcoPredict

A comprehensive ecological prediction system that uses machine learning to predict environmental risks and species distribution patterns.

## Overview

EcoPredict combines climate data, land use information, and species occurrence data to predict ecological risks and generate actionable insights for environmental conservation and management.

## Features

- **Multi-source Data Integration**: Climate, land use, and species occurrence data
- **Machine Learning Models**: Random Forest, XGBoost, and Linear Regression
- **GIS Integration**: Spatial analysis and risk zone mapping
- **Interactive Dashboard**: Real-time visualization and analysis
- **REST API**: Programmatic access to predictions
- **Docker Support**: Containerized deployment

## Project Structure

```
EcoPredict/
├── config/           # Configuration files
├── data/            # Data storage (raw, processed, external)
├── models/          # Trained models and metrics
├── notebooks/       # Jupyter notebooks for analysis
├── scripts/         # Utility scripts
├── src/            # Source code
│   ├── api/        # REST API
│   ├── dashboard/  # Web dashboard
│   ├── gis/        # GIS processing
│   ├── ingestion/  # Data loading
│   ├── models/     # ML models
│   ├── prediction/ # Prediction engine
│   ├── preprocessing/ # Data preprocessing
│   ├── training/   # Model training
│   └── utils/      # Utilities
└── tests/          # Test suite
```

## Quick Start

### Using Docker

```bash
docker-compose up -d
```

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
python scripts/run_pipeline.py
```

3. Start the dashboard:
```bash
python src/dashboard/app.py
```

4. Start the API:
```bash
python src/api/main.py
```

## Configuration

Edit configuration files in the `config/` directory:
- `config.yaml`: General project settings
- `model_params.yaml`: ML model parameters
- `paths.yaml`: Data and output paths

## API Endpoints

- `GET /predict`: Get predictions for coordinates
- `GET /risk-zones`: Get risk zone boundaries
- `GET /heatmap`: Generate risk heatmaps

## License

MIT License - see LICENSE file for details.