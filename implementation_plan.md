# Proje Uygulama Planı: Yangın ve Duman Tespiti

## Hedef
Kamera görüntülerinden (veya fotoğraflardan) ortamda yangın/duman olup olmadığını tespit eden bir Derin Öğrenme (CNN) modeli geliştirmek.

## Veri Seti
- **Kaynak:** Açık kaynaklı "Fire Detection Dataset" (Kaggle veya GitHub üzerinden).
- **Sınıflar:** 2 Sınıf: `Fire` (Yangın) ve `Non-Fire` (Normal/Doğa).
- **Yapı:**
  ```text
  dataset/
    train/
      fire/
      non_fire/
    test/
      fire/
      non_fire/
  ```

## Teknik Yaklaşım

### 1. Veri Ön İşleme (`dataset.py`)
- Resimler sabit boyuta getirilecek (örn. 128x128 veya 224x224 piksel).
- **Data Augmentation:** Eğitim verisini zenginleştirmek için rastgele döndürme, yatay çevirme (flip) uygulanacak. Bu, modelin ezberlemesini (overfitting) engeller.
- Tensor dönüşümü ve Normalizasyon (0-1 arasına çekme) yapılacak.

### 2. Model Mimarisi (`model.py`)
- **Giriş:** 3 kanallı (RGB) resim.
- **Katmanlar:**
  1. Conv2d (3 -> 32 filtre) + ReLU + MaxPool
  2. Conv2d (32 -> 64 filtre) + ReLU + MaxPool
  3. Conv2d (64 -> 128 filtre) + ReLU + MaxPool
  4. Flatten (Düzleştirme)
  5. Dropout (%50 - Ezberlemeyi önlemek için)
  6. FC (Fully Connected) -> Çıkış (2 sınıf: Var/Yok)

### 3. Eğitim (`train.py`)
- **Loss Fonksiyonu:** `CrossEntropyLoss`
- **Optimizasyon:** `Adam` (Learning Rate: 0.001)
- **Süreç:** 10-15 Epoch eğitim. Her epoch sonunda Validation başarısı ölçülecek.

### 4. Arayüz / Test (`predict.py`)
- Kullanıcının verdiği bir resim dosyasını alıp "YANGIN VAR" veya "GÜVENLİ" çıktısı veren bir script.
- Opsiyonel: Klasördeki rastgele resimler üzerinde toplu test.

## Dosya Yapısı Planı
- `data/` (Veri seti buraya indirilecek)
- `src/`
  - `dataset.py`: Veri yükleme işlemleri
  - `model.py`: CNN sınıfı
  - `train.py`: Eğitim döngüsü
  - `utils.py`: Yardımcı fonksiyonlar (grafik çizdirme vb.)
- `main.py`: Projeyi çalıştırmak için ana dosya
- `requirements.txt`: Gerekli kütüphaneler (torch, torchvision, matplotlib, pillow vb.)
