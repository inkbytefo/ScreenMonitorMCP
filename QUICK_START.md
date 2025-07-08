# 🚀 Quick Start Guide - Revolutionary MCP Server

## ⚡ **HIZLI BAŞLANGIÇ**

### **1. Kütüphaneleri Yükle**
```bash
pip install -r requirements.txt
```

### **2. .env Dosyasını Ayarla**
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_OPENAI_MODEL=gpt-4o
API_KEY=your_secret_key
HOST=127.0.0.1
PORT=7777
```

### **3. MCP İstemci Konfigürasyonu**

Claude Desktop veya diğer MCP istemcinizde:

```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": ["C:/Users/tpoyr/OneDrive/Desktop/ScreenMonitorMCP/main.py"],
      "cwd": "C:/Users/tpoyr/OneDrive/Desktop/ScreenMonitorMCP"
    }
  }
}
```

**⚠️ Önemli:** Dosya yolunu kendi proje dizininize göre güncelleyin!

### **4. Test Et**
```bash
# Sunucuyu test et
python main.py

# Devrimsel özellikleri test et
python test_revolutionary_features.py
```

## 🔥 **DEVRIMSEL ARAÇLAR**

### **🔄 Real-Time Monitoring**
```python
# AI'nın sürekli görme özelliği
await start_continuous_monitoring(fps=3)
await get_monitoring_status()
await stop_continuous_monitoring()
```

### **🎯 UI Intelligence**
```python
# UI elementlerini tanı ve etkileşim kur
await analyze_ui_elements()
await smart_click("Kaydet butonuna tıkla")
await extract_text_from_screen()
```

### **🧠 Predictive AI**
```python
# Davranış öğren ve tahmin et
await learn_user_patterns()
await predict_user_intent()
await proactive_assistance()
```

## 🛠️ **SORUN GİDERME**

### **Unicode Hatası (Windows) - ÇÖZÜLDİ ✅**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
Bu hata artık otomatik düzeltiliyor!

### **JSON Hatası**
```json
// ❌ Yanlış - son virgül
{"args": ["path",]}

// ✅ Doğru
{"args": ["path"]}
```

### **Python Path Sorunu**
```json
{
  "command": "C:/Python311/python.exe",
  "args": ["full/path/to/main.py"]
}
```

### **Dependencies Eksik**
```bash
pip install opencv-python numpy structlog pytesseract easyocr pyautogui
```

## 🎯 **İLK DENEME**

MCP istemcinizde şu komutları deneyin:

1. `list_tools()` - Tüm araçları gör
2. `start_continuous_monitoring()` - AI'nın gözlerini aç
3. `analyze_ui_elements()` - Ekranı analiz et
4. `smart_click("close button", dry_run=true)` - Akıllı tıklama test et

## 🚀 **BAŞARILI!**

Artık AI'nız:
- 👁️ Sürekli ekranı izleyebilir
- 🧠 UI elementlerini tanıyabilir  
- 🔮 Davranışlarınızı öğrenebilir
- ⚡ Proaktif yardım edebilir

**Gelecek nesil AI-insan etkileşimi başladı! 🔥**
