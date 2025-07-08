"""
Predictive Intelligence System
Devrimsel özellik: AI'ya kullanıcı davranışlarını öğrenme ve gelecek tahminleri yapma yetisi
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import os
import structlog
import asyncio

logger = structlog.get_logger()

@dataclass
class UserAction:
    """Kullanıcı aksiyonu"""
    timestamp: datetime
    action_type: str  # 'click', 'type', 'scroll', 'app_switch', 'window_change'
    target: str  # Element description or app name
    coordinates: Optional[Tuple[int, int]] = None
    text_content: Optional[str] = None
    app_context: Optional[str] = None
    window_title: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Dictionary'ye çevir"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserAction':
        """Dictionary'den oluştur"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class BehaviorPattern:
    """Davranış kalıbı"""
    pattern_id: str
    pattern_type: str  # 'temporal', 'sequential', 'contextual'
    description: str
    frequency: int
    confidence: float
    last_seen: datetime
    triggers: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    time_patterns: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Dictionary'ye çevir"""
        data = asdict(self)
        data['last_seen'] = self.last_seen.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BehaviorPattern':
        """Dictionary'den oluştur"""
        data['last_seen'] = datetime.fromisoformat(data['last_seen'])
        return cls(**data)

@dataclass
class Prediction:
    """Tahmin"""
    prediction_id: str
    prediction_type: str  # 'next_action', 'workflow', 'assistance'
    description: str
    confidence: float
    suggested_actions: List[str]
    context: Dict[str, Any]
    expires_at: datetime
    
    def is_expired(self) -> bool:
        """Tahmin süresi dolmuş mu?"""
        return datetime.now() > self.expires_at

class PatternRecognizer:
    """Kalıp tanıma sistemi"""
    
    def __init__(self):
        self.temporal_patterns = {}  # Zaman bazlı kalıplar
        self.sequential_patterns = {}  # Sıralı kalıplar
        self.contextual_patterns = {}  # Bağlamsal kalıplar
    
    def analyze_temporal_patterns(self, actions: List[UserAction]) -> List[BehaviorPattern]:
        """Zaman bazlı kalıpları analiz eder"""
        patterns = []
        
        # Saatlik kalıplar
        hourly_actions = defaultdict(list)
        for action in actions:
            hour = action.timestamp.hour
            hourly_actions[hour].append(action)
        
        for hour, hour_actions in hourly_actions.items():
            if len(hour_actions) >= 3:  # En az 3 aksiyon
                action_types = [a.action_type for a in hour_actions]
                most_common = max(set(action_types), key=action_types.count)
                
                pattern = BehaviorPattern(
                    pattern_id=f"temporal_hour_{hour}_{most_common}",
                    pattern_type="temporal",
                    description=f"Saat {hour}:00'da genellikle {most_common} aksiyonu",
                    frequency=len(hour_actions),
                    confidence=min(0.9, len(hour_actions) / 10),
                    last_seen=max(a.timestamp for a in hour_actions),
                    time_patterns={"hour": hour, "action_type": most_common}
                )
                patterns.append(pattern)
        
        return patterns
    
    def analyze_sequential_patterns(self, actions: List[UserAction]) -> List[BehaviorPattern]:
        """Sıralı kalıpları analiz eder"""
        patterns = []
        
        # 2-3 aksiyonluk sekansları bul
        for window_size in [2, 3]:
            sequences = defaultdict(int)
            
            for i in range(len(actions) - window_size + 1):
                sequence = tuple(a.action_type for a in actions[i:i+window_size])
                sequences[sequence] += 1
            
            for sequence, count in sequences.items():
                if count >= 2:  # En az 2 kez tekrarlanmış
                    pattern = BehaviorPattern(
                        pattern_id=f"sequential_{'_'.join(sequence)}",
                        pattern_type="sequential",
                        description=f"Sıralı aksiyon: {' → '.join(sequence)}",
                        frequency=count,
                        confidence=min(0.8, count / 5),
                        last_seen=datetime.now(),
                        actions=list(sequence)
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def analyze_contextual_patterns(self, actions: List[UserAction]) -> List[BehaviorPattern]:
        """Bağlamsal kalıpları analiz eder"""
        patterns = []
        
        # Uygulama bazlı kalıplar
        app_actions = defaultdict(list)
        for action in actions:
            if action.app_context:
                app_actions[action.app_context].append(action)
        
        for app, app_action_list in app_actions.items():
            if len(app_action_list) >= 3:
                action_types = [a.action_type for a in app_action_list]
                most_common = max(set(action_types), key=action_types.count)
                
                pattern = BehaviorPattern(
                    pattern_id=f"contextual_{app}_{most_common}",
                    pattern_type="contextual",
                    description=f"{app} uygulamasında genellikle {most_common}",
                    frequency=len(app_action_list),
                    confidence=min(0.85, len(app_action_list) / 8),
                    last_seen=max(a.timestamp for a in app_action_list),
                    triggers=[app],
                    actions=[most_common]
                )
                patterns.append(pattern)
        
        return patterns

class PredictiveEngine:
    """Tahmin motoru"""
    
    def __init__(self, data_file: str = "user_behavior_data.json"):
        self.data_file = data_file
        self.actions_history: deque = deque(maxlen=1000)  # Son 1000 aksiyon
        self.patterns: List[BehaviorPattern] = []
        self.predictions: List[Prediction] = []
        self.pattern_recognizer = PatternRecognizer()
        
        # Veri dosyasını yükle
        self.load_data()
    
    def record_action(self, action: UserAction):
        """Kullanıcı aksiyonunu kaydet"""
        self.actions_history.append(action)
        logger.info("User action recorded", 
                   action_type=action.action_type,
                   target=action.target)
        
        # Periyodik olarak kalıpları güncelle
        if len(self.actions_history) % 10 == 0:
            self.update_patterns()
        
        # Veriyi kaydet
        self.save_data()
    
    def update_patterns(self):
        """Kalıpları güncelle"""
        try:
            actions_list = list(self.actions_history)
            
            # Farklı kalıp türlerini analiz et
            temporal_patterns = self.pattern_recognizer.analyze_temporal_patterns(actions_list)
            sequential_patterns = self.pattern_recognizer.analyze_sequential_patterns(actions_list)
            contextual_patterns = self.pattern_recognizer.analyze_contextual_patterns(actions_list)
            
            # Mevcut kalıpları güncelle
            all_new_patterns = temporal_patterns + sequential_patterns + contextual_patterns
            
            # Kalıpları birleştir (aynı ID'li olanları güncelle)
            pattern_dict = {p.pattern_id: p for p in self.patterns}
            for new_pattern in all_new_patterns:
                pattern_dict[new_pattern.pattern_id] = new_pattern
            
            self.patterns = list(pattern_dict.values())
            
            logger.info("Patterns updated", 
                       total_patterns=len(self.patterns),
                       temporal=len(temporal_patterns),
                       sequential=len(sequential_patterns),
                       contextual=len(contextual_patterns))
            
        except Exception as e:
            logger.error("Pattern update failed", error=str(e))
    
    def generate_predictions(self, current_context: Dict[str, Any]) -> List[Prediction]:
        """Mevcut bağlama göre tahminler üret"""
        predictions = []
        current_time = datetime.now()
        
        try:
            # Zaman bazlı tahminler
            current_hour = current_time.hour
            for pattern in self.patterns:
                if (pattern.pattern_type == "temporal" and 
                    pattern.time_patterns.get("hour") == current_hour and
                    pattern.confidence > 0.5):
                    
                    prediction = Prediction(
                        prediction_id=f"temporal_{current_hour}_{int(time.time())}",
                        prediction_type="next_action",
                        description=f"Bu saatte genellikle {pattern.time_patterns.get('action_type')} yaparsınız",
                        confidence=pattern.confidence,
                        suggested_actions=[pattern.time_patterns.get('action_type', '')],
                        context={"hour": current_hour, "pattern_id": pattern.pattern_id},
                        expires_at=current_time + timedelta(hours=1)
                    )
                    predictions.append(prediction)
            
            # Bağlamsal tahminler
            current_app = current_context.get("current_app")
            if current_app:
                for pattern in self.patterns:
                    if (pattern.pattern_type == "contextual" and 
                        current_app in pattern.triggers and
                        pattern.confidence > 0.6):
                        
                        prediction = Prediction(
                            prediction_id=f"contextual_{current_app}_{int(time.time())}",
                            prediction_type="workflow",
                            description=f"{current_app} uygulamasında {pattern.actions[0] if pattern.actions else 'bir aksiyon'} yapmanız muhtemel",
                            confidence=pattern.confidence,
                            suggested_actions=pattern.actions,
                            context={"app": current_app, "pattern_id": pattern.pattern_id},
                            expires_at=current_time + timedelta(minutes=30)
                        )
                        predictions.append(prediction)
            
            # Sıralı tahminler (son aksiyonlara göre)
            if len(self.actions_history) >= 2:
                last_actions = [a.action_type for a in list(self.actions_history)[-2:]]
                last_sequence = tuple(last_actions)
                
                for pattern in self.patterns:
                    if (pattern.pattern_type == "sequential" and 
                        pattern.actions and
                        len(pattern.actions) > len(last_sequence) and
                        tuple(pattern.actions[:len(last_sequence)]) == last_sequence):
                        
                        next_action = pattern.actions[len(last_sequence)]
                        prediction = Prediction(
                            prediction_id=f"sequential_{next_action}_{int(time.time())}",
                            prediction_type="next_action",
                            description=f"Sıradaki aksiyon muhtemelen: {next_action}",
                            confidence=pattern.confidence * 0.8,  # Biraz daha düşük güven
                            suggested_actions=[next_action],
                            context={"sequence": last_actions, "pattern_id": pattern.pattern_id},
                            expires_at=current_time + timedelta(minutes=10)
                        )
                        predictions.append(prediction)
            
            # Eski tahminleri temizle
            self.predictions = [p for p in self.predictions if not p.is_expired()]
            
            # Yeni tahminleri ekle
            self.predictions.extend(predictions)
            
            logger.info("Predictions generated", count=len(predictions))
            
        except Exception as e:
            logger.error("Prediction generation failed", error=str(e))
        
        return predictions
    
    def get_proactive_suggestions(self) -> List[Dict[str, Any]]:
        """Proaktif öneriler al"""
        suggestions = []
        
        try:
            current_context = {
                "current_time": datetime.now(),
                "current_app": None  # Bu gerçek uygulamada doldurulacak
            }
            
            # Yeni tahminler üret
            predictions = self.generate_predictions(current_context)
            
            # Yüksek güvenli tahminleri önerilere çevir
            for prediction in predictions:
                if prediction.confidence > 0.7:
                    suggestion = {
                        "type": "suggestion",
                        "message": prediction.description,
                        "actions": prediction.suggested_actions,
                        "confidence": prediction.confidence,
                        "context": prediction.context
                    }
                    suggestions.append(suggestion)
            
            # Kalıp bazlı öneriler
            high_confidence_patterns = [p for p in self.patterns if p.confidence > 0.8]
            if high_confidence_patterns:
                pattern = max(high_confidence_patterns, key=lambda x: x.confidence)
                suggestion = {
                    "type": "pattern_insight",
                    "message": f"En güçlü davranış kalıbınız: {pattern.description}",
                    "confidence": pattern.confidence,
                    "frequency": pattern.frequency
                }
                suggestions.append(suggestion)
            
        except Exception as e:
            logger.error("Proactive suggestions failed", error=str(e))
        
        return suggestions
    
    def get_user_insights(self) -> Dict[str, Any]:
        """Kullanıcı davranış analizi"""
        try:
            insights = {
                "total_actions": len(self.actions_history),
                "total_patterns": len(self.patterns),
                "pattern_breakdown": {
                    "temporal": len([p for p in self.patterns if p.pattern_type == "temporal"]),
                    "sequential": len([p for p in self.patterns if p.pattern_type == "sequential"]),
                    "contextual": len([p for p in self.patterns if p.pattern_type == "contextual"])
                },
                "most_common_actions": {},
                "peak_activity_hours": [],
                "strongest_patterns": []
            }
            
            # En yaygın aksiyonlar
            if self.actions_history:
                action_counts = defaultdict(int)
                for action in self.actions_history:
                    action_counts[action.action_type] += 1
                
                insights["most_common_actions"] = dict(
                    sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                )
                
                # Yoğun aktivite saatleri
                hour_counts = defaultdict(int)
                for action in self.actions_history:
                    hour_counts[action.timestamp.hour] += 1
                
                insights["peak_activity_hours"] = sorted(
                    hour_counts.items(), key=lambda x: x[1], reverse=True
                )[:3]
            
            # En güçlü kalıplar
            insights["strongest_patterns"] = [
                {
                    "description": p.description,
                    "confidence": p.confidence,
                    "frequency": p.frequency,
                    "type": p.pattern_type
                }
                for p in sorted(self.patterns, key=lambda x: x.confidence, reverse=True)[:5]
            ]
            
            return insights
            
        except Exception as e:
            logger.error("User insights generation failed", error=str(e))
            return {"error": str(e)}
    
    def save_data(self):
        """Veriyi dosyaya kaydet"""
        try:
            data = {
                "actions": [action.to_dict() for action in self.actions_history],
                "patterns": [pattern.to_dict() for pattern in self.patterns],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error("Data save failed", error=str(e))
    
    def load_data(self):
        """Veriyi dosyadan yükle"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Aksiyonları yükle
                self.actions_history = deque(
                    [UserAction.from_dict(action_data) for action_data in data.get("actions", [])],
                    maxlen=1000
                )
                
                # Kalıpları yükle
                self.patterns = [
                    BehaviorPattern.from_dict(pattern_data) 
                    for pattern_data in data.get("patterns", [])
                ]
                
                logger.info("Data loaded successfully", 
                           actions=len(self.actions_history),
                           patterns=len(self.patterns))
            
        except Exception as e:
            logger.error("Data load failed", error=str(e))

# Global instance
_predictive_engine: Optional[PredictiveEngine] = None

def get_predictive_engine() -> PredictiveEngine:
    """Global predictive engine instance'ını döndürür"""
    global _predictive_engine
    if _predictive_engine is None:
        _predictive_engine = PredictiveEngine()
    return _predictive_engine
