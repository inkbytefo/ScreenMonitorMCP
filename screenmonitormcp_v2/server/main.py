#!/usr/bin/env python3
"""
ScreenMonitorMCP v2 - Ana Uygulama Giriş Noktası

Çift Kanallı Etkileşimli Analiz Mimarisi ile MCP v3 protokolünü destekler.
"""

from .app import app

# FastAPI uygulamasını dışa aktar
__all__ = ["app"]