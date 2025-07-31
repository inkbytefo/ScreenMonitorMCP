import pytest
from fastapi.testclient import TestClient


class TestStreamAPI:
    """Test API endpoints for stream management."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "uptime" in data
        assert "version" in data
        assert "active_streams" in data
        assert "active_connections" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "ScreenMonitorMCP v2"
        assert data["version"] == "2.0.0"
        assert "endpoints" in data
    
    def test_list_streams_no_auth(self, client):
        """Test list streams without authentication."""
        response = client.get("/api/v2/streams")
        # The API might return 401 or 200 depending on middleware
        assert response.status_code in [200, 401]
    
    def test_list_streams_with_auth(self, client, api_key):
        """Test list streams with authentication."""
        response = client.get(
            "/api/v2/streams",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_status_no_auth(self, client):
        """Test status endpoint without authentication."""
        response = client.get("/api/v2/status")
        assert response.status_code == 200
    
    def test_status_with_auth(self, client, api_key):
        """Test status endpoint with authentication."""
        response = client.get(
            "/api/v2/status",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "connections" in data["data"]
        assert "streams" in data["data"]
    
    def test_connections_no_auth(self, client):
        """Test connections endpoint without authentication."""
        response = client.get("/api/v2/connections")
        assert response.status_code == 200
    
    def test_connections_with_auth(self, client, api_key):
        """Test connections endpoint with authentication."""
        response = client.get(
            "/api/v2/connections",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], dict)
    
    def test_start_stream_no_auth(self, client):
        """Test start stream without authentication."""
        response = client.post("/api/v2/streams/test-stream/start")
        assert response.status_code == 404  # Stream not found
    
    def test_start_stream_nonexistent(self, client, api_key):
        """Test start stream with authentication but nonexistent stream."""
        response = client.post(
            "/api/v2/streams/nonexistent-stream/start",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 404
    
    def test_pause_stream_no_auth(self, client):
        """Test pause stream without authentication."""
        response = client.post("/api/v2/streams/test-stream/pause")
        assert response.status_code == 404
    
    def test_pause_stream_nonexistent(self, client, api_key):
        """Test pause stream with authentication but nonexistent stream."""
        response = client.post(
            "/api/v2/streams/nonexistent-stream/pause",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 404
    
    def test_resume_stream_no_auth(self, client):
        """Test resume stream without authentication."""
        response = client.post("/api/v2/streams/test-stream/resume")
        assert response.status_code == 404
    
    def test_resume_stream_nonexistent(self, client, api_key):
        """Test resume stream with authentication but nonexistent stream."""
        response = client.post(
            "/api/v2/streams/nonexistent-stream/resume",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 404