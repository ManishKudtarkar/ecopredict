.PHONY: help install dev prod test clean lint format docker-build docker-up docker-down deploy logs monitor

# Colors for terminal output
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[0;33m
NC := \033[0m # No Color

PYTHON := python3
DOCKER := docker
DOCKER_COMPOSE := docker-compose

help:
	@echo "$(BLUE)EcoPredict - Ecological Risk Prediction System$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@echo "  make install         Install dependencies"
	@echo "  make dev             Install dev dependencies"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make test            Run unit tests"
	@echo "  make lint            Run linters (flake8, pylint)"
	@echo "  make format          Format code (black, isort)"
	@echo "  make clean           Remove cache and build artifacts"
	@echo ""
	@echo "$(GREEN)Notebooks:$(NC)"
	@echo "  make notebooks-run   Run all notebooks"
	@echo "  make notebooks-export Export notebooks to Python files"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build    Build Docker images"
	@echo "  make docker-up       Start services (docker-compose up)"
	@echo "  make docker-down     Stop services (docker-compose down)"
	@echo "  make docker-logs     View service logs"
	@echo "  make docker-clean    Remove containers and volumes"
	@echo ""
	@echo "$(GREEN)Database:$(NC)"
	@echo "  make db-init         Initialize database"
	@echo "  make db-migrate      Run migrations"
	@echo "  make db-reset        Reset database (dangerous!)"
	@echo ""
	@echo "$(GREEN)Deployment:$(NC)"
	@echo "  make deploy-prod     Deploy to production"
	@echo "  make deploy-staging  Deploy to staging"
	@echo "  make health-check    Run health checks"
	@echo "  make logs            View application logs"
	@echo ""

.DEFAULT_GOAL := help

# Installation targets
install:
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

dev: install
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PYTHON) -m pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Dev environment ready$(NC)"

# Code quality targets
test:
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete$(NC)"

lint:
	@echo "$(BLUE)Running linters...$(NC)"
	$(PYTHON) -m flake8 src/ --max-line-length=100 --statistics
	$(PYTHON) -m pylint src/ --disable=C0111,R0913
	@echo "$(GREEN)✓ Lint checks complete$(NC)"

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	$(PYTHON) -m black src/ tests/ --line-length=100
	$(PYTHON) -m isort src/ tests/ --profile black
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean:
	@echo "$(BLUE)Cleaning cache and build artifacts...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Notebook targets
notebooks-run:
	@echo "$(BLUE)Running Jupyter notebooks...$(NC)"
	jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb
	jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb
	jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb
	jupyter nbconvert --to notebook --execute notebooks/04_results_analysis.ipynb
	@echo "$(GREEN)✓ Notebooks executed$(NC)"

notebooks-export:
	@echo "$(BLUE)Exporting notebooks to Python scripts...$(NC)"
	jupyter nbconvert --to script notebooks/01_data_exploration.ipynb --output-dir=scripts/
	jupyter nbconvert --to script notebooks/02_feature_engineering.ipynb --output-dir=scripts/
	jupyter nbconvert --to script notebooks/03_model_training.ipynb --output-dir=scripts/
	jupyter nbconvert --to script notebooks/04_results_analysis.ipynb --output-dir=scripts/
	@echo "$(GREEN)✓ Notebooks exported$(NC)"

# Docker targets
docker-build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build --no-cache
	@echo "$(GREEN)✓ Images built$(NC)"

docker-up:
	@echo "$(BLUE)Starting services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@sleep 5
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "API Documentation: http://localhost:8000/docs"
	@echo "Dashboard: http://localhost:8501"

docker-down:
	@echo "$(BLUE)Stopping services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs:
	@echo "$(BLUE)Viewing service logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

docker-clean:
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v
	$(DOCKER) system prune -f
	@echo "$(GREEN)✓ Docker cleanup complete$(NC)"

# Database targets
db-init:
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) -c "from scripts.train_model import initialize_db; initialize_db()"
	@echo "$(GREEN)✓ Database initialized$(NC)"

db-migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

db-reset:
	@echo "$(YELLOW)WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) exec postgres dropdb ecopredict; \
		$(DOCKER_COMPOSE) exec postgres createdb ecopredict; \
		$(MAKE) db-init; \
		echo "$(GREEN)✓ Database reset$(NC)"; \
	fi

# Health checks
health-check:
	@echo "$(BLUE)Running health checks...$(NC)"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "API not responding"
	@curl -s http://localhost:8501 >/dev/null && echo "Dashboard is running" || echo "Dashboard not responding"
	@echo "$(GREEN)✓ Health check complete$(NC)"

# Logging
logs:
	@echo "$(BLUE)Viewing application logs...$(NC)"
	tail -f logs/app.log 2>/dev/null || echo "Creating new log file..."
	@echo "$(GREEN)✓ Logs displayed$(NC)"

# Deployment targets
deploy-prod:
	@echo "$(BLUE)Deploying to production...$(NC)"
	@echo "Step 1: Running tests..."
	@$(MAKE) test || (echo "$(RED)Tests failed!$(NC)" && exit 1)
	@echo "Step 2: Building Docker images..."
	@$(MAKE) docker-build
	@echo "Step 3: Starting services..."
	@$(MAKE) docker-up
	@echo "Step 4: Running health checks..."
	@sleep 10
	@$(MAKE) health-check
	@echo "$(GREEN)✓ Deployment complete$(NC)"

deploy-staging:
	@echo "$(BLUE)Deploying to staging...$(NC)"
	@echo "Running with ENVIRONMENT=staging"
	ENVIRONMENT=staging $(MAKE) docker-up
	@echo "$(GREEN)✓ Staging deployment complete$(NC)"

# Development server
dev-server:
	@echo "$(BLUE)Starting development API server...$(NC)"
	$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dev-dashboard:
	@echo "$(BLUE)Starting development dashboard...$(NC)"
	streamlit run src/dashboard/app.py

# Requirements management
requirements-freeze:
	@echo "$(BLUE)Freezing requirements...$(NC)"
	$(PYTHON) -m pip freeze > requirements-frozen.txt
	@echo "$(GREEN)✓ Requirements frozen$(NC)"

requirements-check:
	@echo "$(BLUE)Checking for outdated packages...$(NC)"
	$(PYTHON) -m pip list --outdated

.PHONY: all test
all: clean lint test docker-build docker-up

# Utility targets
version:
	@$(PYTHON) -c "import sys; print(f'Python {sys.version}')"
	@$(DOCKER) --version
	@$(DOCKER_COMPOSE) --version

info:
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "Python: $$(python --version)"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$(docker-compose --version)"
	@echo ""
	@echo "Environment:"
	@if [ -f .env ]; then echo "  .env file: FOUND"; else echo "  .env file: NOT FOUND"; fi
	@if [ -f config/config.yaml ]; then echo "  config.yaml: FOUND"; else echo "  config.yaml: NOT FOUND"; fi
