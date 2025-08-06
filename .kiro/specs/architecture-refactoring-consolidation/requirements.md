# Requirements Document

## Introduction

Bu proje, ScreenMonitorMCP v2 kod tabanındaki ciddi kod tekrarını ve mimari tutarsızlıkları ortadan kaldırarak, her işlevin tek bir yetkili merkezden yönetildiği, "Tek Sorumluluk Prensibi"ne (Single Responsibility Principle) uygun bir yapı oluşturmayı amaçlamaktadır. Mevcut durumda AI servisleri, ekran yakalama mantığı ve konfigürasyon yönetimi birden fazla modülde tekrar etmekte ve bu durum kod sürdürülebilirliğini olumsuz etkilemektedir.

## Requirements

### Requirement 1

**User Story:** Bir geliştirici olarak, AI işlemlerinin tek bir merkezi servisten yönetilmesini istiyorum, böylece kod tekrarı ortadan kalksın ve AI mantığı tutarlı olsun.

#### Acceptance Criteria

1. WHEN AI analizi gerektiğinde THEN sistem sadece core/ai_service.py modülünü kullanmalıdır
2. WHEN core/ai_analyzer.py ve core/ai_vision.py modülleri silindiğinde THEN hiçbir fonksiyonellik kaybı yaşanmamalıdır
3. WHEN uzmanlaşmış AI prompt'ları (detect_ui_elements, assess_system_performance, detect_anomalies) taşındığında THEN bunlar AIService sınıfının metotları olarak çalışmalıdır
4. WHEN stream_analysis_generator fonksiyonu core/streaming.py'ye taşındığında THEN birleştirilmiş ai_service nesnesini kullanmalıdır
5. WHEN diğer modüller AI işlemi yapmak istediğinde THEN sadece core.ai_service.ai_service nesnesini import etmelidirler

### Requirement 2

**User Story:** Bir geliştirici olarak, ekran yakalama işlemlerinin tek bir merkezi modülden yönetilmesini istiyorum, böylece farklı kütüphaneler arası tutarsızlık ortadan kalksın.

#### Acceptance Criteria

1. WHEN ekran yakalama gerektiğinde THEN sistem sadece core/screen_capture.py modülünü (mss kütüphanesi) kullanmalıdır
2. WHEN core/command_handler.py'deki PIL.ImageGrab kullanan metotlar silindiğinde THEN hiçbir fonksiyonellik kaybı yaşanmamalıdır
3. WHEN CommandHandler sınıfı ekran görüntüsü almak istediğinde THEN core.screen_capture.ScreenCapture sınıfından bir nesne kullanmalıdır
4. WHEN ThreadPoolExecutor yapısı korunduğunda THEN performans etkilenmemelidir
5. WHEN core/streaming.py'deki ScreenStreamer sınıfı ekran yakalama yaptığında THEN core.screen_capture.ScreenCapture modülünü kullanmalıdır

### Requirement 3

**User Story:** Bir geliştirici olarak, konfigürasyon ayarlarının tek bir merkezi dosyadan yönetilmesini istiyorum, böylece ayar tutarsızlıkları ortadan kalksın.

#### Acceptance Criteria

1. WHEN konfigürasyon ayarlarına erişim gerektiğinde THEN sistem sadece server/config.py dosyasını kullanmalıdır
2. WHEN core/config.py dosyası silindiğinde THEN hiçbir konfigürasyon ayarı kaybolmamalıdır
3. WHEN core dizini altındaki modüller konfigürasyon istediğinde THEN server.config modülündeki config nesnesini import etmelidirler
4. WHEN Pydantic BaseSettings yapısı korunduğunda THEN environment variable yönetimi çalışmaya devam etmelidir
5. WHEN göreceli import yolları kullanıldığında THEN (from ..server.config import config) doğru çalışmalıdır

### Requirement 4

**User Story:** Bir geliştirici olarak, protokol katmanlarının (MCP ve API) sadece istekleri yönlendirmesini istiyorum, böylece iş mantığı core katmanında merkezileşsin.

#### Acceptance Criteria

1. WHEN mcp_server.py'deki analyze_screen aracı çağrıldığında THEN sadece core katmanındaki ilgili servisi çağırmalıdır
2. WHEN server/routes.py'deki /analyze/screen endpoint'i çağrıldığında THEN aynı core servisini çağırmalıdır
3. WHEN protokol katmanları minimize edildiğinde THEN sadece istek ayrıştırma ve yanıt formatlama yapmalıdırlar
4. WHEN iş mantığı core katmanına taşındığında THEN protokol katmanları sadece köprü görevi görmelidir
5. WHEN kod tekrarı ortadan kaldırıldığında THEN aynı işlevsellik farklı protokollerden erişilebilir olmalıdır

### Requirement 5

**User Story:** Bir geliştirici olarak, gereksiz import ifadelerinin ve bağımlılıkların temizlenmesini istiyorum, böylece kod tabanı temiz ve sürdürülebilir olsun.

#### Acceptance Criteria

1. WHEN silinen modüllerden kalan import ifadeleri tespit edildiğinde THEN bunlar tüm projeden temizlenmelidir
2. WHEN gereksiz bağımlılıklar bulunduğunda THEN bunlar kaldırılmalıdır
3. WHEN isimlendirme tutarsızlıkları tespit edildiğinde THEN proje genelinde tutarlı hale getirilmelidir
4. WHEN iyi tasarlanmış modüller (database_pool.py, performance_monitor.py) korunduğunda THEN merkezileştirilmiş servislerle doğru entegre olmalıdırlar
5. WHEN refactoring tamamlandığında THEN tüm testler başarıyla çalışmalıdır

### Requirement 6

**User Story:** Bir geliştirici olarak, yeniden yapılandırma sonrasında kod tabanının belirli kalite özelliklerine sahip olmasını istiyorum, böylece gelecekteki geliştirmeler daha kolay olsun.

#### Acceptance Criteria

1. WHEN kod tabanı incelendiğinde THEN aynı işi yapan kod blokları bulunmamalıdır (Tekrarsızlık)
2. WHEN temel hizmetler kontrol edildiğinde THEN AI, ekran yakalama ve konfigürasyon tek merkezden yönetilmelidir (Merkezileşme)
3. WHEN yeni özellik ekleme gerektiğinde THEN bu kolay ve anlaşılır olmalıdır (Sürdürülebilirlik)
4. WHEN sistem performansı ölçüldüğünde THEN gereksiz nesne oluşturma ve tutarsız kütüphane kullanımları ortadan kalkmış olmalıdır (Performans)
5. WHEN kod kalitesi değerlendirildiğinde THEN Single Responsibility Principle'a uygun olmalıdır