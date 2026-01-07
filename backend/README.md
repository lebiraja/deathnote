# Life Expectancy Prediction API

Production-ready FastAPI backend for AI-powered life expectancy prediction with health insights and personalized recommendations.

## Features

- 🤖 **ML-Powered Predictions**: Gradient Boosting model with 87% accuracy
- 🔒 **Security**: JWT authentication, rate limiting, CORS protection
- 📊 **Health Insights**: Personalized analysis of risk factors
- 💡 **Recommendations**: Evidence-based health improvement suggestions
- 📈 **Monitoring**: System metrics, model metrics, health checks
- 🐳 **Production-Ready**: Docker support, structured logging, error handling
- ✅ **Tested**: Comprehensive unit and integration tests
- 📚 **API Documentation**: Auto-generated OpenAPI (Swagger) docs

## Tech Stack

- **Framework**: FastAPI 0.115+
- **ML**: scikit-learn 1.5+, pandas, numpy
- **Validation**: Pydantic 2.10+
- **Caching**: Redis 5.2+
- **Testing**: pytest, pytest-asyncio
- **Server**: uvicorn 0.32+
- **Monitoring**: psutil, prometheus-client

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/      # API endpoints
│   │       │   ├── prediction.py
│   │       │   ├── health.py
│   │       │   └── metrics.py
│   │       └── router.py       # API router
│   ├── core/
│   │   ├── config.py           # Configuration
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── logging.py          # Logging setup
│   │   └── security.py         # Security utilities
│   ├── ml/
│   │   └── model_manager.py    # ML model management
│   ├── models/
│   │   ├── prediction.py       # Pydantic models
│   │   ├── health.py
│   │   └── metrics.py
│   ├── services/
│   │   ├── ml_service.py       # ML prediction service
│   │   └── recommendation_service.py
│   └── main.py                 # FastAPI application
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── conftest.py             # Test fixtures
├── models/                     # Trained ML models
├── data/                       # Data files
├── logs/                       # Application logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, for caching)
- Docker (optional, for containerized deployment)

### Local Development

1. **Clone and navigate to backend**:
```bash
cd backend
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Copy ML models to models/ directory**:
```bash
cp ../gradient_boosting_model.pkl models/
cp ../preprocessor.pkl models/
cp ../scaler.pkl models/
```

6. **Run the server**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

7. **Access the API**:
- API: http://localhost:8000
- Docs: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

### Docker Deployment

1. **Build and run with Docker Compose**:
```bash
docker-compose up -d
```

2. **Check logs**:
```bash
docker-compose logs -f api
```

3. **Stop services**:
```bash
docker-compose down
```

## API Endpoints

### Predictions

**POST** `/api/v1/predictions/predict`
- Predict life expectancy with health insights
- Request body: Health metrics (age, BMI, cholesterol, etc.)
- Response: Prediction, confidence, insights, recommendations

### Health Checks

**GET** `/api/v1/health`
- Basic health check

**GET** `/api/v1/health/detailed`
- Detailed health with system and model status

### Metrics

**GET** `/api/v1/metrics/system`
- System resource metrics (CPU, memory, disk)

**GET** `/api/v1/metrics/model`
- ML model information and metrics

## Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/predictions/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "Male",
    "bmi": 24.5,
    "cholesterol": 180.0,
    "blood_pressure": "Normal",
    "diabetes": false,
    "hypertension": false,
    "heart_disease": false,
    "asthma": false,
    "smoking_status": "Never",
    "physical_activity": "High",
    "diet": "Good"
  }'
```

## Example Response

```json
{
  "success": true,
  "prediction": 78.5,
  "confidence": 0.87,
  "profile": {
    "bmi": 24.5,
    "bmi_category": "Normal",
    "cholesterol": 180.0,
    "cholesterol_status": "Desirable",
    "activity": "High",
    "smoking": "Never",
    "risk_factors": 0
  },
  "insights": [
    {
      "category": "Body Composition",
      "message": "Your BMI is in the healthy range",
      "severity": "info"
    }
  ],
  "recommendations": [
    {
      "title": "Maintain Healthy Lifestyle",
      "description": "Continue your excellent health practices",
      "priority": "medium"
    }
  ],
  "model_version": "1.0.0",
  "timestamp": "2025-01-09T12:00:00Z"
}
```

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

Run specific test file:
```bash
pytest tests/integration/test_api_endpoints.py -v
```

## Configuration

Environment variables (see `.env.example`):

- `ENVIRONMENT`: dev/staging/production
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)
- `CORS_ORIGINS`: Allowed CORS origins
- `MODEL_PATH`: Path to ML model file
- `REDIS_URL`: Redis connection URL (optional)

## Monitoring

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### System Metrics
```bash
curl http://localhost:8000/api/v1/metrics/system
```

### Model Metrics
```bash
curl http://localhost:8000/api/v1/metrics/model
```

## Security Features

- ✅ **Input Validation**: Pydantic models with field validators
- ✅ **Rate Limiting**: 10 requests/minute per IP
- ✅ **CORS Protection**: Configurable origins
- ✅ **Error Handling**: No sensitive data in error responses
- ✅ **Logging**: Structured JSON logs with rotation
- ✅ **JWT Ready**: Security utilities for authentication
- ✅ **Non-root Docker**: Runs as unprivileged user

## Production Deployment

### Using Docker

1. Build production image:
```bash
docker build -t life-expectancy-api:latest .
```

2. Run container:
```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models:ro \
  --name life-expectancy-api \
  life-expectancy-api:latest
```

### Using Docker Compose

```bash
docker-compose -f docker-compose.yml up -d
```

### Environment Setup

For production:
1. Set `ENVIRONMENT=production`
2. Set `DEBUG=false`
3. Configure proper `CORS_ORIGINS`
4. Use Redis for caching
5. Set up proper logging directory
6. Use secrets management for sensitive data

## Development

### Code Quality

Format code:
```bash
black app tests
```

Lint code:
```bash
flake8 app tests
```

Type check:
```bash
mypy app
```

### Adding New Features

1. Create Pydantic models in `app/models/`
2. Implement service logic in `app/services/`
3. Add endpoints in `app/api/v1/endpoints/`
4. Update router in `app/api/v1/router.py`
5. Write tests in `tests/`

## Troubleshooting

### Model not loading
- Ensure model files are in `models/` directory
- Check file permissions
- Verify model file format (pickle/joblib)

### Redis connection failed
- Start Redis: `docker-compose up redis -d`
- Check `REDIS_URL` in `.env`

### Port already in use
- Change port: `uvicorn app.main:app --port 8001`
- Or kill process using port 8000

## License

MIT License - see LICENSE file

## Support

For issues and questions:
- Create an issue on GitHub
- Check API docs: http://localhost:8000/api/docs
- Review logs: `tail -f logs/app.log`
