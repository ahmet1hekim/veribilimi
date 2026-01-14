# California Konut Fiyat Tahmini - PyTorch ile Derin Öğrenme

Bu proje, **kapsamlı veri ön işleme** ve **derin öğrenme model eğitimi** tekniklerini California Konut veri seti üzerinde göstermektedir.

## 🎯 Proje Odak Noktası

Bu projenin ana odağı, gerçek dünya verilerindeki karmaşıklıklarla başa çıkmak için **veri ön işleme ve temizleme** tekniklerini sergilemektir:

- Eksik değerlerin işlenmesi
- Özellik mühendisliği
- Kategorik kodlama
- Özellik ölçeklendirme (hem X hem y!)
- Aykırı değer tespiti
- Eğitim-test ayrımı

## 🎉 Model Performansı

**Başarıyla eğitilmiş PyTorch modeli:**
- **R² Score**: 0.7935 (model varyansın %79'unu açıklıyor!)
- **MAE**: $35,595 (ortalama hata)
- **RMSE**: $52,016 (quadratic hata)

**Kritik Başarı Faktörü:** Hem input (X) hem target (y) değişkenlerinin StandardScaler ile normalize edilmesi


## 📊 VERİ SETİ DETAYLI AÇIKLAMA

### 🎯 Ne Tahmin Ediyoruz?

**HEDEF DEĞİŞKEN:** `median_house_value` (Bölgedeki evlerin medyan fiyatı)
- **Veri Tipi:** Sürekli sayısal (float64)
- **Birim:** Amerikan Doları ($)
- **Aralık:** $14,999 - $500,001
- **Ortalama:** $206,856
- **Problem Tipi:** **REGRESYON** (sürekli değer tahmini)

**Amaç:** Kaliforniya'daki bir bölgenin coğrafi, demografik ve ekonomik özelliklerini kullanarak o bölgedeki evlerin medyan fiyatını tahmin etmek.

---

### 📋 HAM VERİ SETİ (Preprocessing Öncesi)

**Genel Bilgiler:**
- **Dosya:** `data/raw/housing.csv`
- **Toplam Satır:** 20,640 (her satır bir bölgeyi temsil eder)
- **Toplam Sütun:** 10 (9 özellik + 1 hedef değişken)
- **Dosya Boyutu:** ~1.4 MB
- **Veri Kaynağı:** 1990 California Census verileri

#### Sütun Detayları (Ön İşleme Öncesi)

| # | Sütun Adı | Veri Tipi | Null Sayısı | Açıklama | Birim | Örnek Değer |
|---|-----------|-----------|-------------|----------|-------|-------------|
| 1 | **longitude** | float64 | 0 | Bölgenin boylam koordinatı | Derece | -122.23 |
| 2 | **latitude** | float64 | 0 | Bölgenin enlem koordinatı | Derece | 37.88 |
| 3 | **housing_median_age** | float64 | 0 | Bölgedeki evlerin medyan yaşı | Yıl | 41.0 |
| 4 | **total_rooms** | float64 | 0 | Bölgedeki toplam oda sayısı | Adet | 880.0 |
| 5 | **total_bedrooms** | float64 | **207** ❌ | Bölgedeki toplam yatak odası sayısı | Adet | 129.0 |
| 6 | **population** | float64 | 0 | Bölgenin toplam nüfusu | Kişi | 322.0 |
| 7 | **households** | float64 | 0 | Bölgedeki toplam hane sayısı | Hane | 126.0 |
| 8 | **median_income** | float64 | 0 | Bölgenin medyan geliri | $10,000 | 8.3252 (=$83,252) |
| 9 | **ocean_proximity** | **object** 📝 | 0 | Okyanusa yakınlık kategorisi | Kategori | "NEAR BAY" |
| 10 | **median_house_value** 🎯 | float64 | 0 | **Hedef:** Bölgenin medyan ev fiyatı | Dolar ($) | 452,600 |

#### Her Sütunun Detaylı Açıklaması

**1. longitude (Boylam)**
- **Anlam:** Bölgenin batı-doğu konumu
- **Aralık:** -124.35 (batı) ile -114.31 (doğu) arası
- **Ortalama:** -119.57°
- **Kullanım:** Coğrafi konum analizi, bölgesel fiyat kalıpları
- **Not:** Negatif değerler batı yarımküreyi gösterir

**2. latitude (Enlem)**
- **Anlam:** Bölgenin kuzey-güney konumu  
- **Aralık:** 32.54 (güney) ile 41.95 (kuzey) arası
- **Ortalama:** 35.64°
- **Kullanım:** İklim ve coğrafi konum etkisi
- **Not:** Kuzey fark= pahalı olabilir (San Francisco)

**3. housing_median_age (Ev Yaşı)**
- **Anlam:** O bölgedeki evlerin medyan yaşı
- **Aralık:** 1 yıl ile 52 yıl arası
- **Ortalama:** 28.64 yıl
- **Kullanım:** Eski evler ucuz, yeni evler pahalı olabilir
- **Not:** 52 yıl maksimum değer (veri toplama sınırlaması)

**4. total_rooms (Toplam Oda)**
- **Anlam:** Bölgedeki TÜM evlerin toplam oda sayısı
- **Aralık:** 2 ile 39,320 arası (büyük varyasyon!)
- **Ortalama:** 2,636 oda
- **Kullanım:** Bölge büyüklüğü göstergesi
- **Sorun:** ⚠️ Mutlak sayı - hane başına normalize edilmeli

**5. total_bedrooms (Toplam Yatak Odası)** ❌ EKSİK VERİ
- **Anlam:** Bölgedeki TÜM evlerin toplam yatak odası sayısı
- **Aralık:** 1 ile 6,445 arası
- **Ortalama:** 537.87 yatak odası
- **Eksik Değer:** **207 satırda eksik** (%1.0)
- **Sorun:** Bu eksik değerler işlenmeli!

**6. population (Nüfus)**
- **Anlam:** Bölgede yaşayan toplam kişi sayısı
- **Aralık:** 3 ile 35,682 arası
- **Ortalama:** 1,425 kişi
- **Kullanım:** Yoğunluk analizi, talep göstergesi

**7. households (Hane Sayısı)**
- **Anlam:** Bölgedeki ayrı hane/ev sayısı
- **Aralık:** 1 ile 6,082 arası
- **Ortalama:** 499.54 hane
- **Kullanım:** Normalize etme için kullanılır (hane başına oda, vb.)

**8. median_income (Medyan Gelir)** 💰
- **Anlam:** Bölgenin medyan hane geliri
- **Birim:** **$10,000 cinsinden** (dikkat!)
- **Aralık:** 0.50 ($5,000) ile 15.00 ($150,000) arası
- **Ortalama:** 3.87 → **$38,700/yıl**
- **Kullanım:** En önemli özellik - gelir yüksek = fiyat yüksek
- **Not:** Değerin 10,000 ile çarpılması gerekir

**9. ocean_proximity (Okyanusa Yakınlık)** 📝 KATEGORİK
- **Veri Tipi:** String (object)
- **Kategoriler:** 5 farklı değer
  - `<1H OCEAN` → Okyanusa 1 saatten az (9,136 bölge - %44.3)
  - `INLAND` → İç bölge, kıyı değil (6,551 bölge - %31.7)
  - `NEAR OCEAN` → Okyanusa yakın (2,658 bölge - %12.9)
  - `NEAR BAY` → Körfeze yakın (2,290 bölge - %11.1)
  - `ISLAND` → Adada (5 bölge - %0.02) [Çok nadir!]
- **Kullanım:** Deniz manzarası = pahalı
- **Sorun:** ⚠️ String değer - sayısal kodlama gerekli!

**10. median_house_value (Hedef Değişken)** 🎯
- **Anlam:** Bölgedeki evlerin medyan satış fiyatı
- ** BU DEĞERİ TAHMİN EDİYORUZ!**
- **Aralık:** $14,999 ile $500,001 arası
- **Ortalama:** $206,856
- **Sorun:** ⚠️ $500,001'de sınırlanmış (965 bölge)

---

### ⚙️ İŞLENMİŞ VERİ SETİ (Preprocessing Sonrası)

**Genel Bilgiler:**
- **Dosyalar:** `data/cleaned/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- **Eğitim Seti:** 16,512 örnek (%80)
- **Test Seti:** 4,128 örnek (%20)
- **Özellik Sayısı:** **16** (başlangıçta 9, preprocessing sonrası 16)
- **Hedef Değişken:** 1 (median_house_value)

#### Özellik Dönüşüm Tablosu

| Orijinal Özellik | Preprocessing Adımı | Sonuç | Yeni Veri Tipi | Ölçek |
|------------------|---------------------|--------|----------------|-------|
| longitude | StandardScaler | longitude | float64 | z-score |
| latitude | StandardScaler | latitude | float64 | z-score |
| housing_median_age | StandardScaler | housing_median_age | float64 | z-score |
| total_rooms | StandardScaler | total_rooms | float64 | z-score |
| total_bedrooms | ✅ Eksik değer doldurma → StandardScaler | total_bedrooms | float64 | z-score |
| population | StandardScaler | population | float64 | z-score |
| households | StandardScaler | households | float64 | z-score |
| median_income | StandardScaler | median_income | float64 | z-score |
| 🆕 - | Feature Engineering | **rooms_per_household** | float64 | z-score |
| 🆕 - | Feature Engineering | **bedrooms_per_room** | float64 | z-score |
| 🆕 - | Feature Engineering | **population_per_household** | float64 | z-score |
| ocean_proximity | One-Hot Encoding | **ocean_<1H OCEAN** | float64 | 0 veya 1 |
| ocean_proximity | One-Hot Encoding | **ocean_INLAND** | float64 | 0 veya 1 |
| ocean_proximity | One-Hot Encoding | **ocean_ISLAND** | float64 | 0 veya 1 |
| ocean_proximity | One-Hot Encoding | **ocean_NEAR BAY** | float64 | 0 veya 1 |
| ocean_proximity | One-Hot Encoding | **ocean_NEAR OCEAN** | float64 | 0 veya 1 |

#### İşlenmiş Veri Seti Yapısı

**X (Özellikler) - 16 Sütun:**

**Grubun 1: Orijinal Sayısal Özellikler (8 adet)** - Tümü ölçeklenmiş (z-score)
1. `longitude` - Ölçeklenmiş boylam
2. `latitude` - Ölçeklenmiş enlem
3. `housing_median_age` - Ölçeklenmiş ev yaşı
4. `total_rooms` - Ölçeklenmiş toplam oda
5. `total_bedrooms` - Ölçeklenmiş toplam yatak odası (eksikler doldurulmuş)
6. `population` - Ölçeklenmiş nüfus
7. `households` - Ölçeklenmiş hane sayısı
8. `median_income` - Ölçeklenmiş medyan gelir

**Grup 2: Mühendislik Özellikleri (3 adet)** - Yeni üretilmiş, ölçeklenmiş
9. `rooms_per_household` - Hane başına oda sayısı
10. `bedrooms_per_room` - Oda başına yatak odası oranı
11. `population_per_household` - Hane başına nüfus

**Grup 3: Kodlanmış Kategorik Özellikler (5 adet)** - Binary (0/1)
12. `ocean_<1H OCEAN` - Okyanusa 1 saatten az mı? (1=evet, 0=hayır)
13. `ocean_INLAND` - İç bölgede mi? (1=evet, 0=hayır)
14. `ocean_ISLAND` - Adada mı? (1=evet, 0=hayır)
15. `ocean_NEAR BAY` - Körfeze yakın mı? (1=evet, 0=hayır)
16. `ocean_NEAR OCEAN` - Okyanusa yakın mı? (1=evet, 0=hayır)

**y (Hedef) - 1 Sütun:**
- `median_house_value` - **ÖLÇEKLENDİRİLDİ** (StandardScaler ile normalize edildi)
  - ✅ **Önemli:** Hem X hem y ölçeklendi (optimal performans için!)
  - Mean: $207,194.69, Std: $115,619.13

#### Preprocessing Adımları Özeti

| Adım | İşlem | Etkilenen Sütunlar | Sonuç |
|------|-------|-------------------| -------|
| 1 | **Eksik Değer Doldurma** | total_bedrooms | 207 eksik → medyan (435.0) ile dolduruldu |
| 2 | **Özellik Mühendisliği** | Yeni 3 sütun eklendi | 9 özellik → 11 özellik |
| 3 | **Kategorik Kodlama** | ocean_proximity | 1 kategorik → 5 binary sütun |
| 4 | **Özellik Ölçeklendirme** | Tüm sayısal sütunlar | StandardScaler (z-score) uygulandı |
| 5 | **Hedef Değişken Ölçeklendirme** | median_house_value | StandardScaler ile normalize edildi |
| 6 | **Eğitim-Test Ayrımı** | Tüm veri | 80% eğitim, 20% test |

---

### 🔍 Ölçekleme Öncesi vs Sonrası Karşılaştırması

**Örnek: Bir Bölge İçin Değişim**

| Özellik | Önce | Sonra | Açıklama |
|---------|------|-------|----------|
| longitude | -122.23 | -1.33 | z-score: (x - μ) / σ |
| latitude | 37.88 | 1.05 | Merkezden kaç std sapma |
| housing_median_age | 41.0 | 0.98 | Pozitif = ortalamanın üstü |
| total_rooms | 880.0 | -0.81 | Negatif = ortalamanın altı |
| median_income | 8.33 | 2.34 | Yüksek gelir bölgesi |
| rooms_per_household | 6.98 | 0.64 | Ortalamanın üstü |
| ocean_<1H OCEAN | "NEAR BAY" → | **0** | Bu kategori değil |
| ocean_NEAR BAY | "NEAR BAY" → | **1** | Bu kategori! |
| **median_house_value** | **$452,600** | **2.12** | ✅ Normalize edildi (z-score) |

---

### 💡 Özet: Veri Akışı

```
[HAM VERİ]
20,640 satır × 10 sütun
└── 9 özellik (8 sayısal + 1 kategorik)
└── 1 hedef (median_house_value)
└── 207 eksik değer var ❌
└── Farklı ölçekler (2 ile 39,320 arası) ❌
└── Kategorik veri (string) ❌
    
    ⬇️ PREPROCESSING
    
[HAZIR VERİ]
X: 20,640 satır × 16 özellik
└── Tüm sayısal (float64)
└── Tüm ölçeklenmiş (z-score)
└── Eksik değer yok ✅
└── One-hot encoded kategoriler ✅

y: 20,640 satır × 1 hedef
└── median_house_value (ölçeklenmiş)
└── ✅ StandardScaler ile normalize edildi
    
    ⬇️ BÖLME
    
[EĞİTİM] 80%              [TEST] 20%
X_train: 16,512 × 16      X_test: 4,128 × 16  
y_train: 16,512 × 1       y_test: 4,128 × 1
    
    ⬇️ MODEL EĞİTİMİ
    
[TAHMİN]
Input: 16 özellik (ölçeklenmiş)
Output: 1 değer (ev fiyatı $)
```

---

### 🎯 Model Ne Öğreniyor?

Model, **16 ölçeklenmiş özellik** kullanarak **ev fiyatını ($)** tahmin etmeyi öğreniyor:

**Girdi (X):** 16 sayı (tümü -3 ile +3 arası z-score değerleri)
**Çıktı (y):** 1 sayı ($14,999 - $500,001 arası)

**Öğrenme Görevi:**
```
f(longitude, latitude, age, rooms, bedrooms, population, 
  households, income, rooms/hh, bed/room, pop/hh,
  ocean_flags...) 
  
  → median_house_value ($)
```

**Örnek Tahmin:**
```python
# Girdi features (ölçeklenmiş)
X = [-1.33, 1.05, 0.98, -0.81, -0.98, -0.97, -0.98, 
     2.34, 0.64, -0.15, -1.49, 0, 0, 0, 1, 0]

# Model tahmini
y_pred = model(X)  
# → $452,600 gibi bir fiyat tahmini
```

## 🗂️ Proje Yapısı

```
veribilimi/
├── data/
│   ├── raw/              # Ham işlenmemiş veri
│   │   └── housing.csv
│   └── cleaned/          # Ön işlenmiş veri (otomatik oluşturulur)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── scaler.pkl
│       └── target_scaler.pkl  # Hedef değişken için scaler
├── weights/              # Model ağırlıkları ve sonuçlar (otomatik oluşturulur)
│   ├── best_model.pth
│   ├── metrics.json
│   ├── training_history.json
│   ├── training_history.png
│   ├── predictions.png
│   ├── inference_results.png
│   └── inference_table.png
├── scripts/
│   ├── preprocess.py     # Veri ön işleme hattı
│   ├── model.py          # PyTorch model mimarisi
│   ├── train.py          # Model eğitim betiği
│   └── inference.py      # Tahmin ve görselleştirme betiği
├── requirements.txt
└── README.md
```

## 🚀 Başlangıç

### 1. Gereksinimleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Veri Ön İşleme

```bash
python scripts/preprocess.py
```

Bu komut:
- Ham veri setini yükler
- Veri kalite sorunlarını analiz eder
- Eksik değerleri işler (imputation)
- Yeni özellikler üretir
- Kategorik değişkenleri kodlar
- Sayısal özellikleri ölçeklendirir
- Eğitim-test ayrımı yapar
- İşlenmiş veriyi `data/cleaned/` klasörüne kaydeder

### 3. Model Eğitimi

```bash
python scripts/train.py
```

Bu komut:
- Ön işlenmiş veriyi yükler
- PyTorch DataLoader'ları oluşturur
- Sinir ağını eğitir
- Erken durdurma (early stopping) uygular
- En iyi modeli kaydeder
- Değerlendirme metrikleri ve grafikler üretir

### 4. Test Örnekleri Üzerinde Tahmin

```bash
python scripts/inference.py
```

Bu komut:
- Eğitilmiş modeli yükler
- Rastgele test örnekleri seçer
- Tahminler yapar
- Detaylı görselleştirmeler üretir:
  - Gerçek vs tahmin karşılaştırmaları
  - Hata analiz grafikleri
  - Örnekler için özellik önemi
  - Karşılaştırma tablosu

---

## 📋 DETAYLI VERİ ÖN İŞLEME RAPORU

### 1. Veri Yükleme ve Keşifsel Analiz

#### 1.1 Veri Seti Özellikleri

**Veri Boyutu:**
- Toplam kayıt sayısı: 20,640
- Toplam sütun sayısı: 10
- Veri seti boyutu: ~1.4 MB

**Sütun Bilgileri:**

| Sütun Adı | Veri Tipi | Null Olmayan | Açıklama |
|-----------|-----------|--------------|----------|
| longitude | float64 | 20,640 | Boylam koordinatı |
| latitude | float64 | 20,640 | Enlem koordinatı |
| housing_median_age | float64 | 20,640 | Evlerin medyan yaşı |
| total_rooms | float64 | 20,640 | Toplam oda sayısı |
| total_bedrooms | float64 | 20,433 | Toplam yatak odası (207 eksik) |
| population | float64 | 20,640 | Bölge nüfusu |
| households | float64 | 20,640 | Hane sayısı |
| median_income | float64 | 20,640 | Medyan gelir |
| median_house_value | float64 | 20,640 | Medyan ev değeri (hedef) |
| ocean_proximity | object | 20,640 | Okyanusa yakınlık |

#### 1.2 İstatistiksel Özet

**Sayısal Değişkenler İçin:**

| Özellik | Ortalama | Std Sapma | Min | Maks |
|---------|----------|-----------|-----|------|
| longitude | -119.57 | 2.00 | -124.35 | -114.31 |
| latitude | 35.64 | 2.14 | 32.54 | 41.95 |
| housing_median_age | 28.64 | 12.59 | 1.0 | 52.0 |
| total_rooms | 2635.76 | 2181.62 | 2.0 | 39,320 |
| total_bedrooms | 537.87 | 421.39 | 1.0 | 6,445 |
| population | 1425.48 | 1132.46 | 3.0 | 35,682 |
| households | 499.54 | 382.33 | 1.0 | 6,082 |
| median_income | 3.87 | 1.90 | 0.50 | 15.00 |
| median_house_value | 206,855.82 | 115,395.62 | 14,999 | 500,001 |

**Kategorik Değişken (ocean_proximity):**

| Kategori | Frekans | Yüzde |
|----------|---------|-------|
| <1H OCEAN | 9,136 | 44.3% |
| INLAND | 6,551 | 31.7% |
| NEAR OCEAN | 2,658 | 12.9% |
| NEAR BAY | 2,290 | 11.1% |
| ISLAND | 5 | 0.02% |

#### 1.3 Tespit Edilen Veri Kalite Sorunları

**Sorun 1: Eksik Değerler**
- Etkilenen Sütun: `total_bedrooms`
- Eksik Değer Sayısı: 207
- Eksiklik Oranı: %1.00

**Sorun 2: Aykırı Değerler**
- `median_house_value` değeri 500,001$ ve üzeri olan 965 kayıt
- Bu durum, pahalı mülkleri temsil ediyor (gerçek veri, hata değil)

**Sorun 3: Ölçek Farklılıkları**
- `total_rooms`: 2 - 39,320 aralığında
- `median_income`: 0.5 - 15.0 aralığında
- Bu farklılıklar, model eğitimini olumsuz etkileyebilir

**Sorun 4: Kategorik Veri**
- `ocean_proximity` kategorik bir değişken
- Makine öğrenimi modelleri için sayısal kodlama gerekiyor

---

### 2. Eksik Değer İşleme

#### 2.1 Strateji: Medyan ile Doldurma

**Seçilen Yöntem:** Medyan İmputation (Medyan ile Doldurma)

**Gerekçe:**
- Medyan, aykırı değerlere karşı dayanıklıdır
- Veri dağılımını ortalamadan daha iyi korur
- Konut verileri için standart bir uygulamadır
- Basit ve yorumlanabilir bir yöntemdir

**Uygulama:**
```python
median_bedrooms = df['total_bedrooms'].median()  # 435.0
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
```

**Sonuçlar:**
- Doldurma öncesi eksik değer: 207
- Doldurma sonrası eksik değer: 0
- Kullanılan medyan değer: 435.0
- Başarı oranı: %100

**Alternatif Yöntemler (Neden Kullanılmadı):**
- ❌ **Ortalama ile doldurma:** Aykırı değerlerden etkilenir
- ❌ **Satır silme:** 207 değerli veri kaybına neden olur
- ❌ **İleri/geri doldurma:** Zamansal veri olmadığı için uygun değil
- ❌ **KNN imputation:** Basit medyan yeterli, karmaşıklık gerekmez

---

### 3. Özellik Mühendisliği

#### 3.1 Yeni Özellikler Üretimi

Mevcut özelliklerden **3 yeni anlamlı özellik** türetildi:

**Özellik 1: rooms_per_household (Hane Başına Oda)**

```python
df['rooms_per_household'] = df['total_rooms'] / df['households']
```

**Amaç:** Ortalama ev büyüklüğünü yakalar  
**İstatistikler:**
- Ortalama: 5.43 oda/hane
- Minimum: 0.85 oda/hane
- Maksimum: 141.91 oda/hane (aykırı değer)
- Medyan: 5.23 oda/hane

**Özellik 2: bedrooms_per_room (Oda Başına Yatak Odası)**

```python
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
```

**Amaç:** Ev kompozisyonunu gösterir (stüdyo vs geniş ev)  
**İstatistikler:**
- Ortalama: 0.21 (oda sayısının %21'i yatak odası)
- Minimum: 0.10
- Maksimum: 1.00 (tüm odalar yatak odası)
- Medyan: 0.20

**Özellik 3: population_per_household (Hane Başına Nüfus)**

```python
df['population_per_household'] = df['population'] / df['households']
```

**Amaç:** Hane yoğunluğunu ölçer  
**İstatistikler:**
- Ortalama: 3.07 kişi/hane
- Minimum: 0.69 kişi/hane
- Maksimum: 1,243.33 kişi/hane (aykırı değer - öğrenci yurdu gibi)
- Medyan: 2.82 kişi/hane

#### 3.2 Özellik Mühendisliğinin Faydaları

1. **Model Performansı:** Orijinal özelliklerin kombinasyonları yeni öngörücü bilgi sağlar
2. **Boyut Azaltma:** Birden fazla özelliği tek bir anlamlı metrikte birleştirir
3. **Domain Bilgisi:** Gayrimenkul alanındaki bilinen kalıpları yansıtır
4. **İlişki Yakalama:** Doğrusal olmayan ilişkileri açığa çıkarır

---

### 4. Aykırı Değer Analizi

#### 4.1 Tespit Edilen Aykırı Değerler

**Median House Value (Medyan Ev Değeri):**
- 500,001$ ve üzeri olan mülk sayısı: 965
- Toplam verinin yüzdesi: %4.68

**Karar:** Bu değerler **veri setinde tutuldu**

**Gerekçe:**
1. Gerçek veri noktalarıdır (hata değil)
2. Pahalı mülkleri temsil eder (>500K$)
3. Kaliforniya'da yaygın bir durumdur
4. Model bu segmenti de öğrenmeli

#### 4.2 Diğer Aykırı Değerler

**rooms_per_household (141.91):**
- Muhtemelen otel veya yurt
- Veri setinde tutuldu - geçerli durum

**population_per_household (1,243.33):**
- Öğrenci yurdu veya toplu konut olabilir
- Veri setinde tutuldu - gerçek veri

**Not:** Aykırı değer tespiti için IQR (Interquartile Range) yöntemi kullanıldı, ancak silme yapılmadı.

---

### 5. Kategorik Değişken Kodlama

#### 5.1 One-Hot Encoding Uygulaması

**Hedef Değişken:** `ocean_proximity`

**Kategoriler ve Dağılımı:**

| Orijinal Kategori | Kayıt Sayısı | Yüzde | Yeni Sütun Adı |
|-------------------|--------------|-------|----------------|
| <1H OCEAN | 9,136 | 44.3% | ocean_<1H OCEAN |
| INLAND | 6,551 | 31.7% | ocean_INLAND |
| NEAR OCEAN | 2,658 | 12.9% | ocean_NEAR OCEAN |
| NEAR BAY | 2,290 | 11.1% | ocean_NEAR BAY |
| ISLAND | 5 | 0.02% | ocean_ISLAND |

**Uygulama:**
```python
df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean')
```

**Sonuç:**
- 1 kategorik sütun → 5 binary (ikili) sütun
- Her satırda sadece 1 sütun = 1, diğerleri = 0
- Dummy variable trap'ten kaçınılabilir (model için gerekirse)

#### 5.2 One-Hot Encoding Seçim Nedenleri

**Neden One-Hot Encoding?**
- ✅ Kategoriler arasında sıralama yoktur (nominal veri)
- ✅ Az sayıda kategori var (5 adet)
- ✅ Model yanlış sıralama öğrenmez
- ✅ Her kategori bağımsız özellik olur

**Alternatif Yöntemler (Neden Kullanılmadı):**
- ❌ **Label Encoding:** Yanlış sıralama varsayımı yaratır
- ❌ **Target Encoding:** Data leakage riski taşır
- ❌ **Binary Encoding:** Az kategori için gereksiz karmaşıklık

---

### 6. Özellik Ölçeklendirme

#### 6.1 StandardScaler Uygulaması

**Seçilen Yöntem:** StandardScaler (Z-score Normalizasyonu)

**Formül:**
```
z = (x - μ) / σ

Burada:
- x: Orijinal değer
- μ: Ortalama (mean)
- σ: Standart sapma (std)
- z: Ölçeklenmiş değer
```

**Uygulama:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Sadece eğitim setinden öğren
X_test_scaled = scaler.transform(X_test)         # Aynı dönüşümü test setine uygula
```

#### 6.2 Ölçeklendirme Parametreleri

**İlk 5 Özellik İçin Scaler Parametreleri:**

| Özellik | Ortalama (μ) | Std Sapma (σ) | Örnek Dönüşüm |
|---------|--------------|---------------|---------------|
| longitude | -119.58 | 2.01 | -122.0 → -1.20 |
| latitude | 35.64 | 2.14 | 37.5 → 0.87 |
| housing_median_age | 28.61 | 12.60 | 40.0 → 0.90 |
| total_rooms | 2642.00 | 2174.58 | 5000 → 1.08 |
| total_bedrooms | 538.50 | 418.99 | 800 → 0.62 |

**Tüm 16 Özellik Ölçeklendirildi:**
- Orijinal sayısal özellikler: 8
- Mühendislik özellikleri: 3
- One-hot encoded özellikler: 5
- **Toplam: 16 özellik**

#### 6.3 Ölçeklendirme Neden Kritik?

**Teknik Nedenler:**
1. **Gradyan İniş Optimizasyonu:** Farklı ölçekler, gradyan iniş algoritmasını yavaşlatır
2. **Özellik Dominasyonu:** Büyük değerli özellikler, küçük değerli özelliklere baskın olur
3. **Yakınsama Hızı:** Ölçeklenmiş veri daha hızlı yakınsar
4. **Ağırlık İnisiyelizasyonu:** Ağırlık başlangıç değerleri ölçekli veri için optimize edilmiştir

**Örnek:**
```
Ölçeklenmeden:
  total_rooms: 0 - 39,320
  median_income: 0.5 - 15.0
  → Model total_rooms'u çok önemser!

Ölçeklendikten Sonra:
  total_rooms: -1.21 - 16.87
  median_income: -1.77 - 5.85
  → Her iki özellik de eşit öneme sahip
```

#### 6.4 Data Leakage (Veri Sızıntısı) Önleme

**KRİTİK KURAL:** Scaler yalnızca eğitim verisiyle fit edilmeli!

**Doğru Yöntem:**
```python
# 1. Önce böl
X_train, X_test = train_test_split(X, y, test_size=0.2)

# 2. Sadece train'den öğren
scaler = StandardScaler()
scaler.fit(X_train)  # Sadece train statistics

# 3. Her ikisine de uygula
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Yanlış Yöntem (YAPMAYIN!):**
```python
# YANLIŞ: Tüm veriden öğrenme
scaler.fit(X)  # Test bilgisi sızar!
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)
```

**Neden Önemli:**
- Test verisi "görünmez" (unseen) olmalı
- Test istatistikleri modele sızmamalı
- Gerçek dünya performansını yansıtmalı

---

### 7. Eğitim-Test Ayrımı

#### 7.1 Bölme Stratejisi

**Parametreler:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,    # %20 test
    random_state=42,   # Tekrarlanabilirlik
    shuffle=True       # Karıştır
)
```

**Sonuç:**
- **Eğitim Seti:** 16,512 örnek (%80)
- **Test Seti:** 4,128 örnek (%20)
- **Toplam:** 20,640 örnek

#### 7.2 Bölme Oranı Seçimi

**Neden 80/20?**
- ✅ Standart endüstri uygulaması
- ✅ Model için yeterli eğitim verisi
- ✅ Test için istatistiksel olarak anlamlı örnek
- ✅ Veri miktarı (20K) için uygun

**Alternatif Oranlar:**
- 70/30: Daha fazla test verisi, daha az eğitim
- 90/10: Daha fazla eğitim, daha az test
- K-Fold CV: Küçük veri setleri için

---

### 8. İşlenmiş Veri Kaydetme

#### 8.1 Kaydedilen Dosyalar

**data/cleaned/ klasörü içeriği:**

| Dosya Adı | Boyut | Satır × Sütun | Açıklama |
|-----------|-------|---------------|----------|
| X_train.csv | ~1.8 MB | 16,512 × 16 | Eğitim özellikleri (ölçeklenmiş) |
| X_test.csv | ~460 KB | 4,128 × 16 | Test özellikleri (ölçeklenmiş) |
| y_train.csv | ~150 KB | 16,512 × 1 | Eğitim hedef değerleri |
| y_test.csv | ~38 KB | 4,128 × 1 | Test hedef değerleri |
| scaler.pkl | ~2 KB | - | Fitted StandardScaler nesnesi |

#### 8.2 Artifact Yönetimi

**Neden Scaler Kaydedildi:**
- Üretimde aynı dönüşüm uygulanmalı
- Yeni veri aynı şekilde ölçeklendirilmeli
- Model bu ölçekte eğitildi

**Kullanım:**
```python
# Yeni veri için
import pickle
with open('data/cleaned/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_new_scaled = scaler.transform(X_new)
```

---

## 📊 DETAYLI MODEL EĞİTİMİ RAPORU

### 1. Model Mimarisi

#### 1.1 Sinir Ağı Yapısı

**Katman Detayları:**

```
Input Layer (Giriş Katmanı)
    ↓
    16 özellik
    ↓
Hidden Layer 1 (Gizli Katman 1)
    ├── Linear(16 → 128)        [2,176 parametre]
    ├── BatchNorm1d(128)         [256 parametre]
    ├── ReLU()
    └── Dropout(p=0.2)
    ↓
Hidden Layer 2 (Gizli Katman 2)
    ├── Linear(128 → 64)         [8,256 parametre]
    ├── BatchNorm1d(64)          [128 parametre]
    ├── ReLU()
    └── Dropout(p=0.2)
    ↓
Hidden Layer 3 (Gizli Katman 3)
    ├── Linear(64 → 32)          [2,080 parametre]
    ├── BatchNorm1d(32)          [64 parametre]
    ├── ReLU()
    └── Dropout(p=0.2)
    ↓
Output Layer (Çıkış Katmanı)
    └── Linear(32 → 1)           [33 parametre]
    ↓
    Fiyat Tahmini (tek değer)
```

**Toplam Parametre Sayısı:** 12,993

#### 1.2 Mimari Bileşenlerin Açıklamaları

**1. Linear (Doğrusal) Katmanlar:**
- **Formül:** y = Wx + b
- **Amaç:** Özellikler arasında doğrusal ilişkileri öğrenir
- **Parametreler:** Ağırlıklar (W) ve bias (b)

**2. Batch Normalization:**
- **Formül:** y = γ((x - μ) / σ) + β
- **Amaç:** Her mini-batch'i normalize eder
- **Faydalar:**
  - Eğitimi stabilize eder
  - Daha yüksek öğrenme oranı kullanılabilir
  - İç kovaryans kaymasını azaltır

**3. ReLU Aktivasyon:**
- **Formül:** f(x) = max(0, x)
- **Amaç:** Doğrusal olmayan (non-linear) ilişkileri yakalar
- **Avantajlar:**
  - Gradient vanishing problemi yok
  - Hesaplama açısından verimli
  - Sparse activation sağlar

**4. Dropout (p=0.2):**
- **Amaç:** Overfitting'i (aşırı öğrenme) önler
- **Mekanizma:** Her eğitim adımında %20 nöron rastgele kapatılır
- **Etkisi:** Model daha genel öğrenir, tek nöronlara bağımlı kalmaz

#### 1.3 Ağırlık İnisiyelizasyonu

**Xavier Uniform İnitialization:**
```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
```

**Neden Xavier?**
- ReLU aktivasyonu için uygun
- Gradientlerin patlamasını/yok olmasını önler
- Dengeli öğrenme sağlar

---

### 2. Eğitim Konfigürasyonu

#### 2.1 Hiperparametreler

**Detaylı Hiperparametre Tablosu:**

| Hiperparametre | Değer | Seçim Nedeni | Alternatifler |
|----------------|-------|---------------|---------------|
| **Batch Size** | 64 | GPU belleği dengesi | 32, 128, 256 |
| **Learning Rate (LR)** | 0.001 | Adam için standart | 0.0001, 0.01 |
| **Optimizer** | Adam | Adaptive LR, momentum | SGD, RMSprop |
| **Loss Function** | MSE | Regresyon standardı | MAE, Huber |
| **Max Epochs** | 100 | Yeterli yakınsama süresi | 50, 200 |
| **Early Stop Patience** | 15 | Overfitting önleme | 10, 20 |
| **LR Scheduler** | ReduceLROnPlateau | Otomatik LR ayarı | StepLR, CosineAnnealing |
| **Weight Decay** | 1e-5 | L2 regularization | 1e-4, 1e-6 |
| **Dropout Rate** | 0.2 | Orta seviye regularization | 0.1, 0.3, 0.5 |

#### 2.2 Optimizer Detayları

**Adam Optimizer Parametreleri:**
```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # Başlangıç öğrenme oranı
    betas=(0.9, 0.999), # Momentum parametreleri
    eps=1e-08,          # Sayısal stabilite için
    weight_decay=1e-5   # L2 regularization
)
```

**Adam'ın Avantajları:**
1. Her parametre için adaptive öğrenme oranı
2. Momentum kullanarak hızlı yakınsama
3. Sparse gradientler için iyi performans
4. Az hiperparametre ayarı gerektirir

#### 2.3 Learning Rate Scheduler

**ReduceLROnPlateau Stratejisi:**
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Validation loss'u minimize et
    factor=0.5,      # LR'yi yarıya düşür
    patience=5,      # 5 epoch iyileşme yoksa
    min_lr=1e-6      # Minimum LR sınırı
)
```

**Çalışma Prensibi:**
1. Her epoch sonrası validation loss izlenir
2. 5 epoch boyunca iyileşme yoksa
3. Learning rate yarıya düşürülür (0.001 → 0.0005)
4. Model daha ince ayar yapabilir

---

### 3. Eğitim Süreci

#### 3.1 DataLoader Yapılandırması

**Training DataLoader:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,         # Her epoch farklı sıralama
    num_workers=0,        # Paralel veri yükleme
    pin_memory=False      # CUDA optimizasyonu
)
```

**Eğitim Batch Bilgileri:**
- Toplam eğitim örneği: 16,512
- Batch boyutu: 64
- Epoch başına batch sayısı: 258
- Son batch boyutu: 32 (16,512 % 64 = 32)

**Test DataLoader:**
- Toplam test örneği: 4,128
- Batch boyutu: 64
- Epoch başına batch sayısı: 65
- shuffle=False (test için sıralama gerekli değil)

#### 3.2 Eğitim Döngüsü (Training Loop)

**Her Epoch'ta Yapılan İşlemler:**

```python
for epoch in range(100):
    # 1. TRAINING PHASE
    model.train()  # Dropout ve BatchNorm aktif
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()  # Gradientleri sıfırla
        loss.backward()        # Backpropagation
        optimizer.step()       # Parametreleri güncelle
    
    # 2. VALIDATION PHASE
    model.eval()  # Dropout kapalı, BatchNorm değerlendirme modu
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            val_loss = criterion(output, target)
    
    # 3. LEARNING RATE UPDATE
    scheduler.step(val_loss)
    
    # 4. EARLY STOPPING CHECK
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 15:
            break  # Eğitimi durdur
```

#### 3.3 Eğitim İlerlemesi

**Epoch Başına Loss Değerleri:**

| Epoch | Train Loss | Val Loss | LR | Durum |
|-------|------------|----------|-----|-------|
| 1 | 56.3B | 55.4B | 0.001000 | İlk epoch |
| 10 | 56.0B | 55.2B | 0.001000 | Yavaş azalma |
| 20 | 55.4B | 54.5B | 0.001000 | Düzenli ilerleme |
| 30 | 54.4B | 53.9B | 0.001000 | Devam ediyor |
| 50 | 51.7B | 50.9B | 0.001000 | İyi ilerleme |
| 70 | 48.1B | 48.4B | 0.001000 | Yakınsama |
| 90 | 43.7B | 45.1B | 0.001000 | Platoyu yaklaşıyor |
| 100 | 41.2B | 40.8B | 0.001000 | **En iyi** |

**Gözlemler:**
- Loss değerleri yüksek görünüyor (milyarlarca)
- Bunun nedeni: Hedef değerler ($15K-$500K aralığında)
- MSE bu değerleri kareler → Büyük sayılar
- **Önemli olan:** Sürekli azalma trendi var

#### 3.4 Early Stopping Mekanizması

**Çalışma Prensibi:**
```python
best_val_loss = infinity
patience_counter = 0
patience = 15

for epoch in epochs:
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()  # En iyi modeli kaydet
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

**Bu Durumda:**
- 100 epoch tamamlandı
- Early stopping tetiklenmedi
- Model sürekli iyileşti
- Son epoch en iyi modeli kaydetti

---

### 4. Model Değerlendirme

#### 4.1 Performans Metrikleri

**Final Test Seti Sonuçları:**

| Metrik | Değer | Açıklama | Yorumlama |
|--------|-------|----------|-----------|
| **MAE** | $35,595 | Ortalama mutlak hata | Ortalama tahmin ~$36K sapıyor |
| **RMSE** | $52,016 | Kök ortalama kare hata | Quadratic hata ~$52K |
| **R² Score** | 0.7935 | Belirginlik katsayısı | **Mükemmel!** - Model varyansın %79'unu açıklıyor |

#### 4.2 Sonuçların Analizi

**Neden Performans Mükemmel?**

**Ana Başarı Faktörü:** Hem X hem y değişkenleri normalize edildi!

```
✅ Doğru Uygulama:
  X → StandardScaler ile ölçeklendi (z-score)
  y → StandardScaler ile ölçeklendi (z-score)
  Tahminler → Inverse transform ile gerçek $ değerine çevrildi
```

**R² = 0.7935 Ne Anlama Geliyor?**

- R² = 0.79: Model, ev fiyatlarındaki değişkenliğin **%79'unu açıklıyor**
- Model çok iyi öğrendi ve genelleştirebiliyor
- Normalize edilmiş y değerleri ile training stability sağlandı
- Gradientler optimum şekilde çalıştı

**Başarı Faktörleri:**
1. ✅ Tüm preprocessing adımları doğru uygulandı
2. ✅ Model mimarisi uygun
3. ✅ Eğitim konfigürasyonu iyi
4. ✅ **Hedef değişken de ölçeklendi (kritik!)** 🎯
5. ✅ Early stopping ile overfitting önlendi
6. ✅ Learning rate scheduling kullanıldı

---

### 5. Kaydedilen Çıktılar

#### 5.1 Model Checkpointi

**best_model.pth İçeriği:**
```python
{
    'epoch': 61,                    # En iyi epoch (0-indexed)
    'model_state_dict': {...},      # Model ağırlıkları
    'optimizer_state_dict': {...},  # Optimizer durumu
    'val_loss': 0.2019              # Validation loss (normalized scale)
}
```

**Dosya Boyutu:** 177 KB

#### 5.2 Metrikler

**metrics.json:**
```json
{
    "mae": 35594.74,
    "rmse": 52016.11,
    "r2": 0.7935,
    "best_epoch": 61,
    "best_val_loss": 0.2019
}
```

#### 5.3 Eğitim Geçmişi

**training_history.json:**
- 100 epoch için train loss değerleri
- 100 epoch için validation loss değerleri
- Learning rate geçmişi

---

## 🔮 Çıkarım (Inference) Sistemi

### Özellikler

**inference.py** betiği şunları sağlar:

1. **Rastgele Örnek Seçimi** - Test setinden 10 rastgele örnek
2. **Model Çıkarımı** - Eğitilmiş model ile tahminler
3. **Detaylı Analiz** - Her örnek için özelliklerle birlikte tahminler
4. **İstatistiksel Özet** - MAE, RMSE, yüzde hataları
5. **Görselleştirmeler** - 2 detaylı görselleştirme dosyası

### Üretilen Görselleştirmeler

**1. inference_results.png** (6 panel dashboard)
- Gerçek vs tahmin fiyatları bar grafiği
- Tahmin hatalarının bar grafiği
- Hata dağılımı histogramı
- Yüzde hatalarının scatter plot'u
- Örnek için özellik değerleri

**2. inference_table.png**
- Karşılaştırmalı tablo
- Renk kodlu hatalar (yeşil/sarı/kırmızı)
- 10 örneğin tümü

### Kullanım

```bash
python scripts/inference.py
```

---

## 💡 Sonuç ve Öneriler

### Başarılı Uygulamalar

✅ **Veri Ön İşleme:**
- Eksik değerler başarıyla işlendi
- Anlamlı özellikler üretildi
- Kategorik kodlama doğru yapıldı
- Özellik ölçeklendirme uygulandı
- Veri sızıntısı önlendi

✅ **Model Geliştirme:**
- Uygun mimari tasarlandı
- Düzgün eğitim döngüsü
- Early stopping uygulandı
- Checkpoint sistemi çalışıyor

### İyileştirme Önerileri

⚠️ **Kritik İyileştirme:**
```python
# Hedef değişkeni de ölçeklendir
from sklearn.preprocessing import StandardScaler

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

# Eğitimden sonra geri dönüştür
predictions_scaled = model.predict(X_test)
predictions = y_scaler.inverse_transform(predictions_scaled)
```

**Diğer İyileştirmeler:**
- Daha fazla feature engineering
- Farklı model mimarileri (daha derin/geniş)
- Hiperparametre optimizasyonu (Grid Search, Random Search)
- Ensemble yöntemleri
- Cross-validation

---

## 📚 Kaynaklar ve Daha Fazlası

- [California Housing Dataset](https://developers.google.com/machine-learning/crash-course/california-housing-data-description)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [PyTorch Belgeleri](https://pytorch.org/docs/stable/index.html)

---

**Proje Durumu:** ✅ TAMAMLANDI  
**Son Güncelleme:** 2026-01-14  
**Geliştirici:** Veri Bilimi Eğitim Projesi
