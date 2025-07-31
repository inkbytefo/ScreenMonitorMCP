# Testing Guide for ScreenMonitorMCP v2

This directory contains unit and integration tests for the ScreenMonitorMCP v2 application.

## Running Tests

### Method 1: Using the test runner script
```bash
python run_tests.py
```

### Method 2: Using pytest directly
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run specific test class
pytest tests/test_api.py::TestStreamManager -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Method 3: Using Python
```bash
python -m pytest tests/ -v
```

## Test Structure

- `test_api.py`: Unit tests for API endpoints and StreamManager
- `test_integration.py`: Integration tests for the complete application
- `conftest.py`: Test configuration and fixtures

## Test Categories

1. **API Tests**: Test individual API endpoints
2. **Unit Tests**: Test core functionality like StreamManager
3. **Integration Tests**: Test complete workflows
4. **Security Tests**: Test API key validation

## Test Dependencies

All test dependencies are already included in `requirements.txt`:
- pytest
- pytest-asyncio
- httpx

## Writing New Tests

When adding new tests:

1. Follow the existing naming conventions (`test_*`)
2. Use the provided fixtures in `conftest.py`
3. Mock external dependencies when possible
4. Add docstrings explaining what each test does
5. Test both success and error cases

## Environment Setup

Tests use a mock API key (`test-api-key-123`) to avoid requiring real credentials. The test suite automatically mocks the configuration to use this key.