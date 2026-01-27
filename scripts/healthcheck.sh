#!/bin/bash
# Health check script for EcoPredict services

set -e

# Check API health
echo "Checking API health..."
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
if [ "$API_STATUS" = "200" ]; then
    echo "✓ API is healthy (HTTP $API_STATUS)"
else
    echo "✗ API health check failed (HTTP $API_STATUS)"
    exit 1
fi

# Check Dashboard health
echo "Checking Dashboard..."
DASHBOARD_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 2>/dev/null || echo "000")
if [ "$DASHBOARD_STATUS" = "200" ]; then
    echo "✓ Dashboard is running (HTTP $DASHBOARD_STATUS)"
else
    echo "✗ Dashboard health check failed (HTTP $DASHBOARD_STATUS)"
    exit 1
fi

# Check Database connectivity
echo "Checking Database..."
if command -v pg_isready &> /dev/null; then
    if pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER 2>/dev/null; then
        echo "✓ PostgreSQL database is accessible"
    else
        echo "✗ PostgreSQL database is not accessible"
        exit 1
    fi
else
    echo "⚠ pg_isready not available, skipping database check"
fi

# Check Prometheus
echo "Checking Prometheus..."
PROM_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9090 2>/dev/null || echo "000")
if [ "$PROM_STATUS" = "200" ]; then
    echo "✓ Prometheus is running (HTTP $PROM_STATUS)"
else
    echo "⚠ Prometheus health check failed (HTTP $PROM_STATUS)"
fi

echo ""
echo "✓ All health checks passed!"
echo ""
echo "Services accessible at:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Dashboard: http://localhost:8501"
echo "  - Prometheus: http://localhost:9090"

exit 0
