# EcoPredict Deployment Guide

## Overview
EcoPredict is a production-ready ecological risk prediction system with REST API, interactive dashboard, and machine learning models for environmental risk assessment.

## Quick Start - Docker (Recommended)

### Prerequisites
- Docker (v20.10+)
- Docker Compose (v1.29+)
- Minimum 4GB RAM
- 2GB disk space

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/yourusername/ecopredict.git
cd ecopredict

# Create environment file from template
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

### 2. Configure Environment Variables
Key variables to set in `.env`:
- `API_HOST` - API server host (default: 0.0.0.0)
- `API_PORT` - API server port (default: 8000)
- `API_WORKERS` - Number of worker processes (default: 4)
- `LOG_LEVEL` - Logging level (INFO, DEBUG, WARNING, ERROR)
- `DATABASE_URL` - Database connection string
- `ALLOWED_HOSTS` - Comma-separated allowed hosts
- `CORS_ORIGINS` - Comma-separated allowed origins

### 3. Deploy with Docker Compose
```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ecopredict-api
docker-compose logs -f ecopredict-dashboard

# Health check
curl http://localhost:8000/health
curl http://localhost:8501
```

### 4. Access Services
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **Dashboard**: http://localhost:8501
- **API Endpoints**:
  - `GET /` - Root endpoint
  - `POST /predict` - Get risk predictions
  - `GET /risk-zones` - Risk zone boundaries
  - `GET /heatmap` - Risk heatmap data

## Manual Installation (Linux/macOS)

### Prerequisites
- Python 3.9+
- pip or conda
- GDAL libraries (for GIS functionality)

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev gcc g++ python3-dev
```

**macOS:**
```bash
brew install gdal
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Application
```bash
# Copy configuration template
cp config/config.yaml.example config/config.yaml

# Edit configuration as needed
nano config/config.yaml
```

### 5. Train or Load Models
```bash
# Run model training
python scripts/train_model.py

# Or copy pre-trained models to models/trained/
```

### 6. Run Application

**API Server:**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Dashboard:**
```bash
streamlit run src/dashboard/app.py
```

## Production Deployment

### AWS ECS
```bash
# Build and push Docker image
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t ecopredict:latest .
docker tag ecopredict:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ecopredict:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ecopredict:latest

# Deploy using CloudFormation or Terraform
```

### Kubernetes
```bash
# Create namespace
kubectl create namespace ecopredict

# Deploy using Helm or kubectl
kubectl apply -f k8s/deployment.yaml -n ecopredict

# Check deployment
kubectl get pods -n ecopredict
kubectl logs -f deployment/ecopredict-api -n ecopredict
```

### Azure Container Instances
```bash
# Push to Azure Container Registry
az acr build --registry ecopredict --image ecopredict:latest .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name ecopredict \
  --image ecopredict.azurecr.io/ecopredict:latest \
  --ports 8000 8501 \
  --environment-variables API_PORT=8000 DASHBOARD_PORT=8501
```

## Monitoring and Logging

### View Logs
```bash
# Docker Compose
docker-compose logs -f --tail=100 ecopredict-api

# System logs
tail -f logs/ecopredict.log

# Structured logs (JSON)
cat logs/ecopredict.log | jq .
```

### Health Monitoring
```bash
# API health
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:9090/api/v1/targets
```

### Performance Monitoring
- Open Prometheus dashboard: http://localhost:9090
- Grafana dashboards (if configured)
- Application logs in `logs/ecopredict.log`

## Database Management

### SQLite (Development)
Default configuration uses SQLite. Database file: `data/ecopredict.db`

```bash
# Backup database
cp data/ecopredict.db data/ecopredict.db.backup

# Restore database
cp data/ecopredict.db.backup data/ecopredict.db
```

### PostgreSQL (Production)
```bash
# Connect to PostgreSQL
psql -h localhost -U ecopredict -d ecopredict

# Create backups
pg_dump -h localhost -U ecopredict ecopredict > backup.sql

# Restore from backup
psql -h localhost -U ecopredict ecopredict < backup.sql
```

## API Usage Examples

### Python
```python
import requests
import json

# Predict risk for a location
url = "http://localhost:8000/predict"
payload = {
    "latitude": 19.5,
    "longitude": 75.9,
    "temperature": 28.5,
    "precipitation": 1.2,
    "forest_cover": 0.45,
    "urban_area": 0.15,
    "species_count": 12,
    "population_density": 85
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

### cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 19.5,
    "longitude": 75.9,
    "temperature": 28.5,
    "precipitation": 1.2,
    "forest_cover": 0.45,
    "urban_area": 0.15,
    "species_count": 12,
    "population_density": 85
  }'
```

### JavaScript/Node.js
```javascript
const axios = require('axios');

const payload = {
    latitude: 19.5,
    longitude: 75.9,
    temperature: 28.5,
    precipitation: 1.2,
    forest_cover: 0.45,
    urban_area: 0.15,
    species_count: 12,
    population_density: 85
};

axios.post('http://localhost:8000/predict', payload)
    .then(response => console.log(response.data))
    .catch(error => console.error(error));
```

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Out of Memory**
```bash
# Increase Docker memory limit
docker update --memory=4g ecopredict-api

# Or modify docker-compose.yml
```

**Model Loading Issues**
```bash
# Check model files exist
ls -la models/trained/

# Verify model compatibility
python scripts/validate_models.py
```

**Database Connection Errors**
```bash
# Check PostgreSQL is running
docker-compose ps

# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Test connection
python -c "import sqlalchemy; engine = sqlalchemy.create_engine('<DATABASE_URL>'); engine.connect()"
```

## Security Considerations

1. **Environment Variables**: Never commit `.env` file to version control
2. **API Keys**: Generate strong API keys if enabled
3. **CORS Configuration**: Restrict to specific origins in production
4. **Database**: Use strong passwords for database users
5. **Logging**: Ensure sensitive data is not logged
6. **Network**: Use VPC/private networks for internal communication
7. **SSL/TLS**: Configure HTTPS with valid certificates

## Performance Optimization

1. **Caching**: Enable prediction caching in config
2. **Workers**: Increase `API_WORKERS` based on CPU cores
3. **Database**: Use connection pooling (configured in SQLAlchemy)
4. **Load Balancing**: Use nginx/HAProxy in front of API
5. **CDN**: Serve static assets from CDN

## Backup and Recovery

### Automated Backups
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup database
docker-compose exec -T postgres pg_dump -U ecopredict ecopredict > $BACKUP_DIR/db.sql

# Backup models
cp -r models $BACKUP_DIR/

# Backup data
cp -r data $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

### Disaster Recovery
```bash
# Restore from backup
docker-compose down
docker volume rm ecopredict_postgres_data
docker-compose up -d

# Restore database
docker-compose exec -T postgres psql -U ecopredict ecopredict < backup.sql

# Restore models and data
cp -r backup_dir/models/* models/
cp -r backup_dir/data/* data/
```

## Support and Updates

- **Documentation**: https://github.com/yourusername/ecopredict/docs
- **Issues**: https://github.com/yourusername/ecopredict/issues
- **Releases**: Check GitHub releases for updates
- **Community**: Join discussions on GitHub

## License
This project is licensed under the MIT License - see LICENSE file for details.
