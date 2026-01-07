# Backend Architecture - Life Expectancy Prediction API

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (React Frontend, Mobile Apps, External Services)               │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ HTTPS/REST
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│  - Rate Limiting                                                 │
│  - CORS                                                          │
│  - Authentication/Authorization                                  │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              API Routes (v1, v2)                         │   │
│  │  /api/v1/predict, /health, /metrics, /docs             │   │
│  └────────────┬────────────────────────────────────────────┘   │
│               │                                                   │
│  ┌────────────▼────────────────────────────────────────────┐   │
│  │                 Controllers                              │   │
│  │  - PredictionController                                  │   │
│  │  - HealthController                                      │   │
│  │  - MetricsController                                     │   │
│  └────────────┬────────────────────────────────────────────┘   │
│               │                                                   │
│  ┌────────────▼────────────────────────────────────────────┐   │
│  │                  Services                                │   │
│  │  - MLService (model loading, prediction)                │   │
│  │  - PreprocessingService                                  │   │
│  │  - RecommendationService                                 │   │
│  │  - CacheService                                          │   │
│  └────────────┬────────────────────────────────────────────┘   │
│               │                                                   │
│  ┌────────────▼────────────────────────────────────────────┐   │
│  │              ML Pipeline                                 │   │
│  │  - DataLoader                                            │   │
│  │  - Preprocessor                                          │   │
│  │  - ModelManager                                          │   │
│  └────────────┬────────────────────────────────────────────┘   │
│               │                                                   │
└───────────────┼───────────────────────────────────────────────────┘
                │
    ┌───────────┴───────────┬──────────────┬────────────────┐
    │                       │              │                │
    ▼                       ▼              ▼                ▼
┌─────────┐         ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Redis  │         │ Database │    │  Models  │    │   Logs   │
│ (Cache) │         │(Optional)│    │  (.pkl)  │    │  (File)  │
└─────────┘         └──────────┘    └──────────┘    └──────────┘
```

## 📁 Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry point
│   ├── config.py                    # Configuration management
│   │
│   ├── api/                         # API layer
│   │   ├── __init__.py
│   │   ├── deps.py                  # Dependencies (DB, auth, etc.)
│   │   └── v1/                      # API version 1
│   │       ├── __init__.py
│   │       ├── endpoints/
│   │       │   ├── __init__.py
│   │       │   ├── prediction.py    # Prediction endpoints
│   │       │   ├── health.py        # Health check endpoints
│   │       │   └── metrics.py       # Metrics endpoints
│   │       └── router.py            # V1 router
│   │
│   ├── core/                        # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration classes
│   │   ├── logging.py               # Logging configuration
│   │   ├── security.py              # Security utilities
│   │   └── exceptions.py            # Custom exceptions
│   │
│   ├── models/                      # Pydantic models (schemas)
│   │   ├── __init__.py
│   │   ├── prediction.py            # Prediction request/response
│   │   ├── health.py                # Health check models
│   │   └── metrics.py               # Metrics models
│   │
│   ├── services/                    # Business logic
│   │   ├── __init__.py
│   │   ├── ml_service.py            # ML prediction service
│   │   ├── preprocessing_service.py # Data preprocessing
│   │   ├── recommendation_service.py# Health recommendations
│   │   └── cache_service.py         # Caching service
│   │
│   ├── ml/                          # Machine learning pipeline
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Data loading
│   │   ├── preprocessing.py         # Feature preprocessing
│   │   ├── model_manager.py         # Model loading/management
│   │   ├── trainer.py               # Model training
│   │   └── evaluator.py             # Model evaluation
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── validators.py            # Custom validators
│       ├── helpers.py               # Helper functions
│       └── constants.py             # Constants
│
├── tests/                           # Testing
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── unit/
│   │   ├── test_ml_service.py
│   │   ├── test_preprocessing.py
│   │   └── test_validators.py
│   ├── integration/
│   │   └── test_api_endpoints.py
│   └── e2e/
│       └── test_prediction_flow.py
│
├── models/                          # Trained ML models
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl
│   └── preprocessor.pkl
│
├── data/                            # Data files
│   └── life-expectancy.csv
│
├── logs/                            # Application logs
│   └── app.log
│
├── scripts/                         # Utility scripts
│   ├── train_model.py               # Train models
│   ├── generate_dataset.py          # Generate synthetic data
│   └── setup.sh                     # Setup script
│
├── .env.example                     # Environment variables template
├── .gitignore
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies
├── pyproject.toml                   # Python project config
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker compose
├── README.md                        # Documentation
└── ARCHITECTURE.md                  # This file
```

## 🔄 Request Flow

```
1. Client Request
   ↓
2. API Gateway (CORS, Rate Limiting, Auth)
   ↓
3. FastAPI Router → Endpoint
   ↓
4. Request Validation (Pydantic)
   ↓
5. Controller (orchestration)
   ↓
6. Service Layer (business logic)
   ↓
7. ML Pipeline (prediction)
   ↓
8. Response Formatting
   ↓
9. Client Response
```

## 🔧 Component Responsibilities

### API Layer (`app/api/`)
- **Purpose**: Handle HTTP requests/responses
- **Responsibilities**:
  - Route definition
  - Request validation
  - Response serialization
  - API versioning

### Core (`app/core/`)
- **Purpose**: Core application functionality
- **Responsibilities**:
  - Configuration management
  - Logging setup
  - Security utilities
  - Custom exceptions

### Models (`app/models/`)
- **Purpose**: Data validation and serialization
- **Responsibilities**:
  - Pydantic schemas
  - Request/response models
  - Data validation rules

### Services (`app/services/`)
- **Purpose**: Business logic implementation
- **Responsibilities**:
  - ML model predictions
  - Data preprocessing
  - Recommendation generation
  - Caching logic

### ML Pipeline (`app/ml/`)
- **Purpose**: Machine learning operations
- **Responsibilities**:
  - Model training
  - Model loading
  - Feature preprocessing
  - Model evaluation

## 🔐 Security Measures

1. **Input Validation**: Pydantic models with strict validation
2. **Rate Limiting**: slowapi for endpoint rate limiting
3. **CORS**: Configured CORS middleware
4. **Environment Variables**: Secure configuration management
5. **Error Handling**: No sensitive data in error responses
6. **Logging**: Secure logging without sensitive data

## 📊 Monitoring & Observability

1. **Health Checks**: `/health` endpoint
2. **Metrics**: `/metrics` endpoint (Prometheus format)
3. **Logging**: Structured logging with rotation
4. **Performance Tracking**: Request timing middleware

## 🚀 Deployment Architecture

### Development
```
docker-compose up
├── FastAPI (port 8000)
├── Redis (port 6379)
└── PostgreSQL (optional, port 5432)
```

### Production
```
Load Balancer (CloudFlare/AWS ALB)
├── FastAPI Instance 1
├── FastAPI Instance 2
├── FastAPI Instance 3
└── FastAPI Instance 4
    ↓
Redis Cluster
    ↓
PostgreSQL (Primary + Replicas)
```

## 📈 Scalability

1. **Horizontal Scaling**: Stateless API design
2. **Caching**: Redis for prediction caching
3. **Async Operations**: FastAPI async endpoints
4. **Connection Pooling**: Database connection pooling
5. **Model Loading**: Lazy loading with LRU cache

## 🔄 CI/CD Pipeline

```
Git Push
  ↓
GitHub Actions
  ├── Run Tests
  ├── Run Linters
  ├── Security Scan
  └── Build Docker Image
      ↓
  Push to Registry
      ↓
  Deploy to Staging
      ↓
  Manual Approval
      ↓
  Deploy to Production
```

## 📝 API Versioning Strategy

- **v1**: Current stable API
- **v2**: Future enhancements (backward compatible when possible)
- URL-based versioning: `/api/v1/`, `/api/v2/`
- Deprecated versions marked in docs with sunset dates

## 🎯 Design Principles

1. **SOLID Principles**: Single responsibility, Open-closed, etc.
2. **Clean Architecture**: Separation of concerns
3. **DRY**: Don't Repeat Yourself
4. **KISS**: Keep It Simple, Stupid
5. **YAGNI**: You Aren't Gonna Need It

## 🧪 Testing Strategy

1. **Unit Tests**: Test individual functions (80%+ coverage)
2. **Integration Tests**: Test API endpoints
3. **E2E Tests**: Test complete user flows
4. **Performance Tests**: Load testing with locust
5. **Security Tests**: OWASP testing

## 📚 Documentation

1. **API Docs**: Auto-generated Swagger/OpenAPI at `/docs`
2. **ReDoc**: Alternative docs at `/redoc`
3. **Architecture**: This document
4. **README**: Setup and usage instructions
5. **Code Comments**: Inline documentation

---

**Version**: 1.0  
**Last Updated**: January 2, 2026  
**Maintained By**: Development Team
