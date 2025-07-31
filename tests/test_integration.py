import pytest
from fastapi.testclient import TestClient


class TestIntegration:
    """Integration tests for the ScreenMonitorMCP v2 API."""
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "uptime" in data
        assert "version" in data
        assert "active_streams" in data
        assert "active_connections" in data
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "ScreenMonitorMCP v2"
        assert data["version"] == "2.0.0"
        assert "endpoints" in data
    
    def test_list_streams_authorized(self, client):
        """Test listing streams."""
        response = client.get("/api/v2/streams")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_status_endpoint(self, client):
        """Test status endpoint."""
        response = client.get("/api/v2/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "connections" in data["data"]
        assert "streams" in data["data"]
    
    def test_list_connections(self, client):
        """Test listing connections."""
        response = client.get("/api/v2/connections")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], dict)
    
    def test_start_stream_nonexistent(self, client):
        """Test starting a non-existent stream."""
        response = client.post("/api/v2/streams/nonexistent-stream/start")
        assert response.status_code == 404
    
    def test_pause_stream_nonexistent(self, client):
        """Test pausing a non-existent stream."""
        response = client.post("/api/v2/streams/nonexistent-stream/pause")
        assert response.status_code == 404
    
    def test_resume_stream_nonexistent(self, client):
        """Test resuming a non-existent stream."""
        response = client.post("/api/v2/streams/nonexistent-stream/resume")
        assert response.status_code == 404
    
    def test_api_endpoints_structure(self, client):
        """Test that all expected endpoints exist and return correct structure."""
        endpoints = [
            "/health",
            "/",
            "/api/v2/streams",
            "/api/v2/status",
            "/api/v2/connections"
        ]
        
        for endpoint in endpoints:
            if endpoint == "/health" or endpoint == "/":
                response = client.get(endpoint)
            else:
                response = client.get(endpoint)
            
            assert response.status_code == 200, f"Endpoint {endpoint} should be accessible"
            
            if endpoint != "/":
                data = response.json()
                assert "success" in data
                assert data["success"] is True
    
    def test_stream_endpoints_structure(self, client):
        """Test stream management endpoints."""
        stream_endpoints = [
            "/api/v2/streams/test-stream/start",
            "/api/v2/streams/test-stream/pause",
            "/api/v2/streams/test-stream/resume"
        ]
        
        for endpoint in stream_endpoints:
            response = client.post(endpoint)
            assert response.status_code == 404  # All should return 404 for non-existent streams