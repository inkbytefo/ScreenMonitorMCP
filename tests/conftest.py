import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient

# Set environment variable for test API key
os.environ['SCREENMONITOR_API_KEY'] = 'test-api-key-123'

# Import after setting environment variable
from screenmonitormcp_v2.server.app import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def api_key():
    """Test API key."""
    return "test-api-key-123"