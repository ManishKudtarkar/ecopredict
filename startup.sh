#!/bin/bash
# EcoPredict Production Startup Script

set -e  # Exit on error

echo "================================"
echo "EcoPredict Production Startup"
echo "================================"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
    echo "✓ Environment variables loaded"
else
    echo "✗ .env file not found"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
if [[ "$PYTHON_VERSION" != "3.9" && "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
    echo "✗ Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION available"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models/trained models/metrics outputs logs
echo "✓ Directories created"

# Check if running in Docker
if [ -f "/.dockerenv" ]; then
    echo "✓ Running in Docker container"
else
    # Create virtual environment if not in Docker
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install -r requirements.txt
        echo "✓ Virtual environment created and dependencies installed"
    else
        source venv/bin/activate
        echo "✓ Virtual environment activated"
    fi
fi

# Validate configuration
echo "Validating configuration..."
python3 -c "
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(f'Project: {config[\"project\"][\"name\"]}')
    print(f'Version: {config[\"project\"][\"version\"]}')
    print(f'Environment: {config[\"project\"][\"environment\"]}')
" || exit 1
echo "✓ Configuration valid"

# Check models exist
if [ ! -d "models/trained" ] || [ -z "$(ls -A models/trained)" ]; then
    echo "⚠ No trained models found"
    echo "  Run: python scripts/train_model.py"
else
    echo "✓ Trained models found"
fi

# Initialize database
echo "Initializing database..."
python3 -c "
from src.utils.helpers import get_db
db = get_db()
print('✓ Database initialized')
" || echo "⚠ Database initialization skipped"

# Health check
echo ""
echo "================================"
echo "Starting Services"
echo "================================"

# Determine how to run based on environment
if [ "$RUN_MODE" == "api" ]; then
    echo "Starting API server..."
    python -m uvicorn src.api.main:app \
        --host ${API_HOST:-0.0.0.0} \
        --port ${API_PORT:-8000} \
        --workers ${API_WORKERS:-4} \
        --access-log \
        --log-level ${LOG_LEVEL:-info}
        
elif [ "$RUN_MODE" == "dashboard" ]; then
    echo "Starting Dashboard..."
    streamlit run src/dashboard/app.py \
        --server.port=${DASHBOARD_PORT:-8501} \
        --server.address=0.0.0.0 \
        --logger.level=${LOG_LEVEL:-info}
        
else
    echo "Starting all services with Docker Compose..."
    docker-compose up -d
    echo ""
    echo "================================"
    echo "Services Started Successfully"
    echo "================================"
    echo ""
    echo "API Documentation: http://localhost:8000/docs"
    echo "Dashboard: http://localhost:8501"
    echo "Health Check: curl http://localhost:8000/health"
    echo ""
    echo "View logs:"
    echo "  docker-compose logs -f ecopredict-api"
    echo "  docker-compose logs -f ecopredict-dashboard"
    echo ""
fi

echo "✓ Startup completed successfully"
