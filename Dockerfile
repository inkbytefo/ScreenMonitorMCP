# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxss1 \
    libgconf-2-4 \
    libxcomposite1 \
    libasound2 \
    libatk1.0-0 \
    libdrm2 \
    libgtk-3-0 \
    libgbm1 \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    lsb-release \
    xdg-utils \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r screenmonitor && useradd -r -g screenmonitor screenmonitor

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=screenmonitor:screenmonitor . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/screenshots && \
    chown -R screenmonitor:screenmonitor /app

# Switch to non-root user
USER screenmonitor

# Set up display for headless operation
ENV DISPLAY=:99

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "screenmonitormcp_v2.mcp_main"]

# Development stage
FROM production as development

# Switch back to root for development dependencies
USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER screenmonitor

# Override command for development
CMD ["python", "-m", "screenmonitormcp_v2.mcp_main", "--debug"]