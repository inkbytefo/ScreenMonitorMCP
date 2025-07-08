# 🚀 Revolutionary Screen Monitor MCP Server

[![CI](https://github.com/yourusername/ScreenMonitorMCP/workflows/ScreenMonitorMCP%20CI/badge.svg)](https://github.com/yourusername/ScreenMonitorMCP/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

**DEVRİMSEL** bir Model Context Protocol (MCP) sunucusu! AI'ya **gerçek zamanlı görme yetisi**, **UI zekası** ve **öngörülü davranış öğrenme** gücü kazandırır. Bu sadece ekran yakalama değil - AI'ya dijital dünyanızı gerçekten "görme" ve anlama gücü veriyor!

## 🌟 **NEDEN ScreenMonitorMCP?**

- 🔥 **İlk ve Tek**: Gerçek zamanlı sürekli ekran izleme özelliği
- 🧠 **AI Zekası**: UI elementlerini anlayan ve etkileşim kurabilen AI
- 🔮 **Öngörülü**: Kullanıcı davranışlarını öğrenen ve tahmin eden sistem
- ⚡ **Proaktif**: İhtiyaç duyulmadan önce yardım öneren asistan
- 🎯 **Doğal**: "Kaydet butonuna tıkla" gibi komutları anlayan AI

## 🔥 **DEVRİMSEL ÖZELLİKLER**

### 🔄 **Gerçek Zamanlı Sürekli İzleme**
- **AI'nın Gözleri Hiç Kapanmaz**: 2-5 FPS sürekli ekran izleme
- **Akıllı Değişiklik Algılama**: Küçük, büyük ve kritik değişiklikleri ayırt eder
- **Proaktif Analiz**: AI önemli değişiklikleri otomatik analiz eder
- **Uyarlanabilir Performans**: Akıllı frame rate ayarlaması

### 🎯 **UI Element Zekası**
- **Computer Vision UI Algılama**: Butonları, formları, menüleri otomatik tanır
- **OCR Metin Çıkarma**: Ekranın herhangi bir yerinden metin okur
- **Akıllı Tıklama Sistemi**: "Kaydet butonuna tıkla" gibi doğal dil komutları
- **Etkileşim Haritalaması**: AI tam olarak nerede ve nasıl etkileşim kuracağını bilir

### 🧠 **Öngörülü Zeka**
- **Davranış Öğrenme**: AI kullanım kalıplarınızı ve alışkanlıklarınızı öğrenir
- **Niyet Tahmini**: Bağlama göre sırada ne yapacağınızı tahmin eder
- **Proaktif Yardım**: Siz sormadan önce yardım önerir
- **İş Akışı Optimizasyonu**: Çalışma kalıplarınızda iyileştirmeler önerir

## 🛠️ **DEVRİMSEL MCP ARAÇLARI**

### 🔄 **Real-Time Monitoring Araçları**
- `start_continuous_monitoring()` - AI'nın sürekli görme özelliğini başlatır
- `stop_continuous_monitoring()` - Sürekli izlemeyi durdurur
- `get_monitoring_status()` - Real-time durum bilgisi ve istatistikler
- `get_recent_changes()` - Son algılanan ekran değişiklikleri

### 🎯 **UI Intelligence Araçları**
- `analyze_ui_elements()` - Ekrandaki tüm UI elementlerini tanır ve haritalandırır
- `smart_click()` - Doğal dil komutları ile akıllı tıklama ("Kaydet butonuna tıkla")
- `extract_text_from_screen()` - OCR ile ekrandan metin çıkarma

### 🧠 **Predictive AI Araçları**
- `learn_user_patterns()` - Kullanıcı davranış kalıplarını öğrenir ve analiz eder
- `predict_user_intent()` - Mevcut bağlama göre kullanıcı niyetini tahmin eder
- `proactive_assistance()` - Kullanıcı istemeden önce proaktif yardım önerir
- `record_user_action()` - Kullanıcı aksiyonlarını kaydeder ve öğrenme sistemine besler

### 📸 **Geleneksel Araçlar**
- `capture_and_analyze()` - Ekran yakalama ve AI analizi (geliştirilmiş)
- `list_tools()` - **MCP standardına uygun** tüm araçları listeler (kategorize edilmiş, detaylı bilgiler)

## 🎯 **KULLANIM SENARYOLARI**

### 🔍 **Real-Time Monitoring**
```python
# AI'nın sürekli görme özelliğini başlat
await start_continuous_monitoring(fps=3, change_threshold=0.1)

# Monitoring durumunu kontrol et
status = await get_monitoring_status()

# Son değişiklikleri gör
changes = await get_recent_changes(limit=5)
```

### 🎯 **UI Intelligence**
```python
# Ekrandaki tüm UI elementlerini analiz et
ui_analysis = await analyze_ui_elements()

# Doğal dil ile akıllı tıklama
await smart_click("Kaydet butonuna tıkla")

# Ekrandan metin çıkar
text_data = await extract_text_from_screen()
```

### 🧠 **Predictive AI**
```python
# Kullanıcı davranış kalıplarını öğren
patterns = await learn_user_patterns()

# Kullanıcı niyetini tahmin et
intent = await predict_user_intent()

# Proaktif yardım al
assistance = await proactive_assistance()
```

## 🚀 **KURULUM**

### **1. Proje Dosyalarını Hazırlayın**
```bash
# Proje dizinine gidin
cd ScreenMonitorMCP

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### **2. Çevre Değişkenlerini Yapılandırın**
`.env` dosyasını düzenleyin:
```env
# Server Configuration
HOST=127.0.0.1
PORT=7777
API_KEY=your_secret_key

# AI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_OPENAI_MODEL=gpt-4o
```

### **3. Bağımsız Test (Opsiyonel)**
```bash
# Sunucuyu test edin
python main.py

# Devrimsel özellikleri test edin
python test_revolutionary_features.py
```

## 🔧 **MCP İSTEMCİ KURULUMU**

### **Claude Desktop / MCP İstemcisi Konfigürasyonu**

MCP istemcinizin konfigürasyon dosyasına aşağıdaki JSON'u ekleyin:

#### **🎯 Basit Konfigürasyon (Önerilen)**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": ["/path/to/ScreenMonitorMCP/main.py"],
      "cwd": "/path/to/ScreenMonitorMCP"
    }
  }
}
```

#### **🔧 Gelişmiş Konfigürasyon**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": [
        "/path/to/ScreenMonitorMCP/main.py"
      ],
      "cwd": "/path/to/ScreenMonitorMCP",
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

#### **🛡️ Güvenli Konfigürasyon**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": [
        "/path/to/ScreenMonitorMCP/main.py",
        "--api-key", "your-secret-key"
      ],
      "cwd": "/path/to/ScreenMonitorMCP"
    }
  }
}
```

#### **🪟 Windows Örneği**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": ["C:/path/to/ScreenMonitorMCP/main.py"],
      "cwd": "C:/path/to/ScreenMonitorMCP"
    }
  }
}
```

### **⚠️ Önemli Notlar**

1. **Dosya Yolu**: `/path/to/ScreenMonitorMCP/main.py` yolunu kendi proje dizininize göre güncelleyin
2. **Python Path**: Python'un PATH'de olduğundan emin olun veya tam yol kullanın: `"C:/Python311/python.exe"`
3. **Çalışma Dizini**: `cwd` parametresi `.env` dosyasının doğru okunması için önemlidir
4. **API Anahtarları**: Tüm ayarlar `.env` dosyasından otomatik okunur

## 🧪 **KULLANIM ÖRNEKLERİ**

### **🔄 Real-Time Monitoring Başlatma**
```python
# AI'nın sürekli görme özelliğini başlat
result = await start_continuous_monitoring(
    fps=3,
    change_threshold=0.1,
    smart_detection=True
)

# Monitoring durumunu kontrol et
status = await get_monitoring_status()

# Son değişiklikleri gör
changes = await get_recent_changes(limit=10)

# Monitoring'i durdur
await stop_continuous_monitoring()
```

### **🎯 UI Intelligence Kullanımı**
```python
# Ekrandaki tüm UI elementlerini analiz et
ui_elements = await analyze_ui_elements(
    detect_buttons=True,
    extract_text=True,
    confidence_threshold=0.7
)

# Doğal dil ile akıllı tıklama
await smart_click("Kaydet butonuna tıkla", dry_run=False)

# Belirli bölgeden metin çıkar
text_data = await extract_text_from_screen(
    region={"x": 100, "y": 100, "width": 500, "height": 300}
)
```

### **🧠 Predictive Intelligence**
```python
# Kullanıcı davranış kalıplarını öğren
patterns = await learn_user_patterns()

# Kullanıcı niyetini tahmin et
intent = await predict_user_intent(
    current_context={"current_app": "VSCode"}
)

# Proaktif yardım al
assistance = await proactive_assistance()

# Kullanıcı aksiyonunu kaydet
await record_user_action(
    action_type="click",
    target="save_button",
    app_context="VSCode"
)
```

### **📸 Geleneksel Ekran Yakalama**
```python
# Geliştirilmiş ekran yakalama ve analiz
result = await capture_and_analyze(
    capture_mode="all",
    analysis_prompt="Bu ekranda ne görüyorsun?",
    max_tokens=500
)

# Tüm araçları listele
tools = await list_tools()
```

## 🚀 **DEVRIMSEL YETENEKLER**

Bu MCP sunucusu AI'ya şu yetenekleri kazandırır:

- 👁️ **Sürekli Görme**: AI hiç durmadan ekranı izleyebilir
- 🧠 **Akıllı Anlama**: UI elementlerini tanır ve etkileşim kurar
- 🔮 **Gelecek Tahmini**: Kullanıcı davranışlarını öğrenip tahmin eder
- ⚡ **Proaktif Yardım**: İhtiyaç duyulmadan önce yardım önerir
- 🎯 **Doğal Etkileşim**: "Kaydet butonuna tıkla" gibi komutları anlar

## 🔧 **SORUN GİDERME**

### **Yaygın Sorunlar ve Çözümleri**

1. **Unicode/Encoding Hatası (Windows)**
   ```
   UnicodeEncodeError: 'charmap' codec can't encode character
   ```
   **Çözüm:** ✅ Bu hata düzeltildi! Sunucu otomatik UTF-8 encoding kullanıyor.

2. **JSON Konfigürasyon Hatası**
   ```json
   // ❌ Yanlış
   {
     "command": "python",
     "args": ["path/to/main.py",]  // Son virgül hatalı
   }

   // ✅ Doğru
   {
     "command": "python",
     "args": ["path/to/main.py"]
   }
   ```

3. **Python Path Sorunu**
   ```json
   {
     "command": "C:/Python311/python.exe",  // Tam path kullanın
     "args": ["C:/path/to/ScreenMonitorMCP/main.py"]
   }
   ```

4. **Dependencies Eksik**
   ```bash
   cd ScreenMonitorMCP
   pip install -r requirements.txt
   ```

5. **OCR Sorunları**
   ```bash
   # Tesseract yükleyin (opsiyonel)
   # EasyOCR otomatik yüklenir
   ```

6. **MCP Connection Closed Hatası**
   ```
   MCP error -32000: Connection closed
   ```
   **Çözüm:** Dosya yollarını kontrol edin ve `cwd` parametresini ekleyin.

## 📝 **LİSANS**

Bu proje MIT Lisansı altında lisanslanmıştır.

---

**🚀 AI'ya gerçek "göz" kazandıran devrimsel MCP sunucusu!**
**🔥 Gelecek nesil AI-insan etkileşimi burada başlıyor!**
