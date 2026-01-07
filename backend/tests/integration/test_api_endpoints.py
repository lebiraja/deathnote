"""
Integration tests for API endpoints.
"""
import pytest
from fastapi import status


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""
    
    def test_predict_success(self, client, sample_prediction_input):
        """Test successful prediction."""
        response = client.post(
            "/api/v1/predictions/predict",
            json=sample_prediction_input
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["success"] is True
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] > 0
        assert "confidence" in data
        assert "profile" in data
        assert "insights" in data
        assert "recommendations" in data
    
    def test_predict_high_risk(self, client, sample_high_risk_input):
        """Test prediction with high-risk profile."""
        response = client.post(
            "/api/v1/predictions/predict",
            json=sample_high_risk_input
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # High-risk should have lower prediction
        assert data["prediction"] < 80
        assert len(data["recommendations"]) > 0
        
        # Should have high priority recommendations
        high_priority_recs = [
            r for r in data["recommendations"]
            if r["priority"] == "high"
        ]
        assert len(high_priority_recs) > 0
    
    def test_predict_invalid_age(self, client, sample_prediction_input):
        """Test prediction with invalid age."""
        invalid_input = sample_prediction_input.copy()
        invalid_input["age"] = 150
        
        response = client.post(
            "/api/v1/predictions/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_bmi(self, client, sample_prediction_input):
        """Test prediction with invalid BMI."""
        invalid_input = sample_prediction_input.copy()
        invalid_input["bmi"] = 5.0
        
        response = client.post(
            "/api/v1/predictions/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_missing_field(self, client, sample_prediction_input):
        """Test prediction with missing required field."""
        invalid_input = sample_prediction_input.copy()
        del invalid_input["age"]
        
        response = client.post(
            "/api/v1/predictions/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestHealthEndpoint:
    """Tests for health check endpoints."""
    
    def test_basic_health(self, client):
        """Test basic health endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_detailed_health(self, client):
        """Test detailed health endpoint."""
        response = client.get("/api/v1/health/detailed")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "model" in data
        assert "system" in data
        
        # Check model status
        assert data["model"]["status"] in ["healthy", "unhealthy", "unknown"]
        
        # Check system status
        assert data["system"]["status"] in ["healthy", "degraded", "unhealthy"]


class TestMetricsEndpoint:
    """Tests for metrics endpoints."""
    
    def test_system_metrics(self, client):
        """Test system metrics endpoint."""
        response = client.get("/api/v1/metrics/system")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_percent" in data
        assert "timestamp" in data
        
        # Validate metric ranges
        assert 0 <= data["cpu_percent"] <= 100
        assert 0 <= data["memory_percent"] <= 100
        assert 0 <= data["disk_percent"] <= 100
    
    def test_model_metrics(self, client):
        """Test model metrics endpoint."""
        response = client.get("/api/v1/metrics/model")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "model_name" in data
        assert "model_version" in data
        assert "model_type" in data
        assert "n_features" in data
        assert "accuracy" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
