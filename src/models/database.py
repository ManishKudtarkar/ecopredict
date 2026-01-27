"""
Database models and initialization
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()


class PredictionResult(Base):
    """Store prediction results for audit trail"""
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    predicted_risk = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    input_features = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_location', 'latitude', 'longitude'),
        Index('idx_created_at', 'created_at'),
    )


class ModelMetrics(Base):
    """Store model training metrics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, unique=True)
    version = Column(String(50), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    r2_score = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    is_active = Column(Boolean, default=False)
    metrics_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_model_version', 'model_name', 'version'),
    )


class UserFeedback(Base):
    """Store user feedback for model improvement"""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    actual_risk_level = Column(String(20), nullable=False)  # low, medium, high
    predicted_risk_level = Column(String(20), nullable=False)
    feedback_text = Column(String(500), nullable=True)
    user_id = Column(String(100), nullable=True)
    is_helpful = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_feedback_location', 'latitude', 'longitude'),
        Index('idx_feedback_date', 'created_at'),
    )


class AuditLog(Base):
    """Audit log for all API operations"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)  # GET, POST, etc
    user_id = Column(String(100), nullable=True)
    status_code = Column(Integer, nullable=False)
    request_data = Column(JSON, nullable=True)
    response_status = Column(String(20), nullable=False)  # success, error
    error_message = Column(String(500), nullable=True)
    execution_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_endpoint', 'endpoint'),
        Index('idx_user', 'user_id'),
        Index('idx_date', 'created_at'),
    )


def init_db(db: Session):
    """Initialize database tables"""
    Base.metadata.create_all(bind=db.bind)
    print("Database initialized successfully")


def get_or_create_default_metrics(db: Session):
    """Create default model metrics if they don't exist"""
    default_metrics = [
        {
            "model_name": "random_forest",
            "version": "1.0.0",
            "accuracy": 0.847,
            "precision": 0.84,
            "recall": 0.85,
            "f1_score": 0.845,
            "r2_score": 0.847,
            "rmse": 0.0159,
            "mae": 0.0098,
            "is_active": True,
            "metrics_json": {
                "cv_mean": 0.842,
                "cv_std": 0.018,
                "training_samples": 800,
                "test_samples": 200
            }
        },
        {
            "model_name": "xgboost",
            "version": "1.0.0",
            "accuracy": 0.823,
            "precision": 0.82,
            "recall": 0.83,
            "f1_score": 0.825,
            "r2_score": 0.823,
            "rmse": 0.0181,
            "mae": 0.0126,
            "is_active": False,
            "metrics_json": {
                "cv_mean": 0.815,
                "cv_std": 0.022,
                "training_samples": 800,
                "test_samples": 200
            }
        }
    ]
    
    for metric_data in default_metrics:
        existing = db.query(ModelMetrics).filter(
            ModelMetrics.model_name == metric_data["model_name"]
        ).first()
        
        if not existing:
            db.add(ModelMetrics(**metric_data))
    
    db.commit()
    print("Default metrics created successfully")
