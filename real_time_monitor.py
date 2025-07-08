"""
Real-Time Screen Monitoring System
Devrimsel özellik: AI'ya sürekli görme yetisi kazandırma
"""

import asyncio
import threading
import time
import cv2
import numpy as np
import mss
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import structlog

logger = structlog.get_logger()

@dataclass
class ChangeEvent:
    """Ekran değişikliği olayını temsil eder"""
    timestamp: datetime
    change_type: str  # 'minor', 'major', 'critical'
    change_percentage: float
    affected_regions: List[Dict]
    screenshot_base64: Optional[str] = None
    description: str = ""

@dataclass
class MonitoringConfig:
    """Real-time monitoring yapılandırması"""
    fps: int = 2
    change_threshold: float = 0.1  # %10 değişiklik eşiği
    major_change_threshold: float = 0.3  # %30 büyük değişiklik
    critical_change_threshold: float = 0.6  # %60 kritik değişiklik
    focus_regions: List[Dict] = field(default_factory=list)
    smart_detection: bool = True
    save_screenshots: bool = True
    max_history: int = 100

class SmartChangeDetector:
    """Akıllı değişiklik algılama sistemi"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.previous_frame = None
        self.frame_history = []
        self.change_patterns = []
        
    def detect_changes(self, current_frame: np.ndarray) -> ChangeEvent:
        """Mevcut frame ile önceki frame arasındaki değişiklikleri algılar"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return ChangeEvent(
                timestamp=datetime.now(),
                change_type='initial',
                change_percentage=0.0,
                affected_regions=[]
            )
        
        # Frame differencing
        diff = cv2.absdiff(self.previous_frame, current_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold uygula
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Değişiklik yüzdesini hesapla
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        change_percentage = changed_pixels / total_pixels
        
        # Değişiklik bölgelerini bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        affected_regions = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Küçük değişiklikleri filtrele
                x, y, w, h = cv2.boundingRect(contour)
                affected_regions.append({
                    'x': int(x), 'y': int(y), 
                    'width': int(w), 'height': int(h),
                    'area': int(cv2.contourArea(contour))
                })
        
        # Değişiklik tipini belirle
        change_type = self._classify_change(change_percentage, affected_regions)
        
        # Screenshot'ı base64'e çevir (eğer gerekirse)
        screenshot_base64 = None
        if self.config.save_screenshots and change_type != 'minor':
            screenshot_base64 = self._frame_to_base64(current_frame)
        
        # Önceki frame'i güncelle
        self.previous_frame = current_frame.copy()
        
        return ChangeEvent(
            timestamp=datetime.now(),
            change_type=change_type,
            change_percentage=change_percentage,
            affected_regions=affected_regions,
            screenshot_base64=screenshot_base64,
            description=self._generate_description(change_type, affected_regions)
        )
    
    def _classify_change(self, change_percentage: float, regions: List[Dict]) -> str:
        """Değişiklik tipini sınıflandırır"""
        if change_percentage >= self.config.critical_change_threshold:
            return 'critical'
        elif change_percentage >= self.config.major_change_threshold:
            return 'major'
        elif change_percentage >= self.config.change_threshold:
            return 'minor'
        else:
            return 'none'
    
    def _generate_description(self, change_type: str, regions: List[Dict]) -> str:
        """Değişiklik için açıklama oluşturur"""
        if change_type == 'none':
            return "Önemli değişiklik algılanmadı"
        
        region_count = len(regions)
        if region_count == 0:
            return f"{change_type.title()} değişiklik algılandı"
        elif region_count == 1:
            return f"{change_type.title()} değişiklik algılandı (1 bölge)"
        else:
            return f"{change_type.title()} değişiklik algılandı ({region_count} bölge)"
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Frame'i base64 string'e çevirir"""
        _, buffer = cv2.imencode('.png', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

class RealTimeMonitor:
    """Real-time ekran izleme sistemi"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.detector = SmartChangeDetector(config)
        self.is_monitoring = False
        self.monitor_thread = None
        self.change_callbacks: List[Callable] = []
        self.event_history: List[ChangeEvent] = []
        self.stats = {
            'total_frames': 0,
            'changes_detected': 0,
            'start_time': None,
            'last_activity': None
        }
    
    def add_change_callback(self, callback: Callable[[ChangeEvent], None]):
        """Değişiklik algılandığında çağrılacak callback ekler"""
        self.change_callbacks.append(callback)
    
    def start_monitoring(self) -> Dict[str, Any]:
        """Monitoring'i başlatır"""
        if self.is_monitoring:
            return {"status": "already_running", "message": "Monitoring zaten çalışıyor"}
        
        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Real-time monitoring başlatıldı", fps=self.config.fps)
        
        return {
            "status": "started",
            "message": f"Real-time monitoring başlatıldı ({self.config.fps} FPS)",
            "config": {
                "fps": self.config.fps,
                "change_threshold": self.config.change_threshold,
                "smart_detection": self.config.smart_detection
            }
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Monitoring'i durdurur"""
        if not self.is_monitoring:
            return {"status": "not_running", "message": "Monitoring zaten durmuş"}
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        duration = datetime.now() - self.stats['start_time']
        
        logger.info("Real-time monitoring durduruldu", 
                   duration=str(duration),
                   total_frames=self.stats['total_frames'],
                   changes_detected=self.stats['changes_detected'])
        
        return {
            "status": "stopped",
            "message": "Real-time monitoring durduruldu",
            "stats": {
                "duration": str(duration),
                "total_frames": self.stats['total_frames'],
                "changes_detected": self.stats['changes_detected'],
                "avg_fps": self.stats['total_frames'] / duration.total_seconds() if duration.total_seconds() > 0 else 0
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Monitoring durumunu döndürür"""
        return {
            "is_monitoring": self.is_monitoring,
            "stats": self.stats.copy(),
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "change_type": event.change_type,
                    "change_percentage": event.change_percentage,
                    "description": event.description
                }
                for event in self.event_history[-10:]  # Son 10 olay
            ]
        }
    
    def _monitor_loop(self):
        """Ana monitoring döngüsü"""
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Tüm ekranları yakala
            
            while self.is_monitoring:
                try:
                    start_time = time.time()
                    
                    # Ekran görüntüsü yakala
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # Değişiklikleri algıla
                    change_event = self.detector.detect_changes(frame)
                    
                    # İstatistikleri güncelle
                    self.stats['total_frames'] += 1
                    if change_event.change_type != 'none':
                        self.stats['changes_detected'] += 1
                        self.stats['last_activity'] = datetime.now()
                        
                        # Event history'ye ekle
                        self.event_history.append(change_event)
                        if len(self.event_history) > self.config.max_history:
                            self.event_history.pop(0)
                        
                        # Callback'leri çağır
                        for callback in self.change_callbacks:
                            try:
                                callback(change_event)
                            except Exception as e:
                                logger.error("Callback error", error=str(e))
                    
                    # FPS kontrolü
                    elapsed = time.time() - start_time
                    sleep_time = max(0, (1.0 / self.config.fps) - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error("Monitor loop error", error=str(e))
                    time.sleep(1.0)  # Hata durumunda kısa bekle

# Global monitor instance
_global_monitor: Optional[RealTimeMonitor] = None

def get_global_monitor() -> Optional[RealTimeMonitor]:
    """Global monitor instance'ını döndürür"""
    return _global_monitor

def set_global_monitor(monitor: RealTimeMonitor):
    """Global monitor instance'ını ayarlar"""
    global _global_monitor
    _global_monitor = monitor
