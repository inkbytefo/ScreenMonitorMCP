# 🤝 Contributing to ScreenMonitorMCP

ScreenMonitorMCP projesine katkıda bulunduğunuz için teşekkür ederiz! Bu rehber, projeye nasıl katkıda bulunabileceğinizi açıklar.

## 🚀 Katkı Türleri

### 🐛 Bug Reports (Hata Raporları)
- Hataları GitHub Issues'da rapor edin
- Detaylı açıklama ve yeniden üretme adımları ekleyin
- Sistem bilgilerinizi (OS, Python versiyonu) belirtin

### 💡 Feature Requests (Özellik İstekleri)
- Yeni özellik önerilerinizi Issues'da paylaşın
- Özelliğin neden gerekli olduğunu açıklayın
- Mümkünse kullanım senaryoları ekleyin

### 🔧 Code Contributions (Kod Katkıları)
- Fork yapın ve feature branch oluşturun
- Kod standartlarına uyun
- Test ekleyin
- Pull Request açın

## 🛠️ Development Setup

### 1. Repository'yi Fork Edin
```bash
git clone https://github.com/yourusername/ScreenMonitorMCP.git
cd ScreenMonitorMCP
```

### 2. Development Environment Kurun
```bash
# Virtual environment oluşturun
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Dependencies yükleyin
pip install -r requirements.txt
```

### 3. Environment Variables Ayarlayın
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

## 📝 Kod Standartları

### Python Code Style
- PEP 8 standartlarına uyun
- Type hints kullanın
- Docstring'leri ekleyin
- Meaningful variable names kullanın

### Commit Messages
```
feat: add new UI detection algorithm
fix: resolve OCR encoding issue
docs: update installation guide
test: add unit tests for monitoring
```

### Branch Naming
```
feature/smart-click-enhancement
bugfix/ocr-unicode-error
docs/contributing-guide
```

## 🧪 Testing

### Unit Tests Çalıştırın
```bash
# Tüm testleri çalıştır
python -m pytest

# Belirli test dosyası
python test_revolutionary_features.py
```

### Manual Testing
```bash
# Sunucuyu test edin
python main.py

# MCP client ile test edin
# Claude Desktop veya başka MCP client kullanın
```

## 📋 Pull Request Süreci

### 1. Branch Oluşturun
```bash
git checkout -b feature/your-feature-name
```

### 2. Değişikliklerinizi Yapın
- Kod yazın
- Test ekleyin
- Dokümantasyon güncelleyin

### 3. Commit ve Push
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 4. Pull Request Açın
- GitHub'da Pull Request açın
- Detaylı açıklama yazın
- İlgili Issue'ları bağlayın

## 🎯 Katkı Alanları

### 🔥 Öncelikli Alanlar
- **UI Detection**: Yeni UI element algılama algoritmaları
- **OCR Improvements**: Daha iyi metin tanıma
- **Performance**: Monitoring performansı optimizasyonu
- **Cross-platform**: Linux/Mac desteği
- **Documentation**: Türkçe/İngilizce dokümantasyon

### 🧠 AI/ML Katkıları
- Behavior prediction algoritmaları
- Smart detection iyileştirmeleri
- Computer vision optimizasyonları

### 🛠️ Infrastructure
- CI/CD pipeline
- Docker containerization
- Package management

## 📚 Dokümantasyon

### README Güncellemeleri
- Yeni özellikler için kullanım örnekleri
- Installation guide iyileştirmeleri
- Troubleshooting bölümü

### Code Documentation
- Docstring'ler
- Type hints
- Inline comments

## 🐛 Bug Fix Süreci

### 1. Issue Oluşturun
- Hatayı detaylı açıklayın
- Yeniden üretme adımları
- Beklenen vs gerçek davranış

### 2. Fix Geliştirin
- Minimal değişiklik yapın
- Test ekleyin
- Edge case'leri düşünün

### 3. Test Edin
- Unit test yazın
- Manual test yapın
- Regression test

## 🔒 Security

### Güvenlik Açıkları
- Güvenlik açıklarını özel olarak rapor edin
- Public issue açmayın
- Email: security@screenmonitormcp.com

### API Keys ve Secrets
- .env dosyalarını commit etmeyin
- Hardcoded secrets kullanmayın
- .gitignore'u kontrol edin

## 📞 İletişim

### GitHub
- Issues: Hata raporları ve özellik istekleri
- Discussions: Genel tartışmalar
- Pull Requests: Kod katkıları

### Community
- Discord: [Yakında]
- Twitter: [Yakında]

## 🏆 Contributors

Tüm katkıda bulunanlar README'de listelenir ve projeye değerli katkıları için teşekkür edilir.

---

**🚀 Birlikte ScreenMonitorMCP'yi daha da devrimsel hale getirelim!**
