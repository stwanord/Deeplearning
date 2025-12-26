# 🔥 Fire & Smoke Detection with CNN

Bu proje, Derin Öğrenme (Deep Learning) yöntemleri kullanılarak kamera görüntülerinden veya fotoğraflardan **Yangın (Fire)**, **Duman (Smoke)** ve **Normal (Neutral)** durumlarını tespit etmek amacıyla geliştirilmiştir.

## 📋 Proje Hakkında
**Ders:** Derin Öğrenme (Deep Learning) - Dönem Projesi  
**Konu:** Görüntü İşleme ile Yangın ve Duman Tespiti  
**Model:** Convolutional Neural Network (CNN)  
**Framework:** PyTorch  

### 🎯 Amaç
Orman yangınları ve endüstriyel kazalar gibi durumlarda erken uyarı sistemleri hayati önem taşır. Bu proje, görsel verileri analiz ederek insansız bir şekilde yangın tespiti yapabilen bir yapay zeka modeli geliştirmeyi hedefler.

## 📂 Veri Seti (Dataset)
Projede kullanılan veri seti 3 sınıftan oluşmaktadır:
1.  **Fire:** Alev içeren görüntüler.
2.  **Smoke:** Yoğun duman içeren görüntüler.
3.  **Neutral:** Yangın veya duman olmayan doğa/şehir görüntüleri.

Veri seti, eğitim (train) ve test aşamaları için ayrı klasörlerde düzenlenmiştir.

## 🛠 Görevi Çalıştırma

### 1. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Modeli Eğitin
Eğer hazır model yoksa veya yeniden eğitmek isterseniz:
```bash
python src/train.py
```
Bu işlem sonucunda `fire_model.pth` dosyası ve `training_results.png` başarım grafiği oluşacaktır.

### 3. Test ve Demo (Arayüz)
Modeli denemek için web arayüzünü başlatın:
```bash
python src/predict.py
```
Komut çalıştıktan sonra terminalde çıkan linke (örn: `http://127.0.0.1:7860`) tıklayın.

## 📊 Model Başarısı
Model 10 Epoch sonunda **%85+** doğruluk (Accuracy) oranına ulaşmıştır. 
*(Eğitim grafikleri `training_results.png` dosyasında mevcuttur)*

## 🧠 Model Mimarisi
- **Giriş:** 224x224 RGB Resim
- **Katmanlar:**
  - 3 adet Convolutional Blok (Conv2d + ReLU + MaxPool)
  - Flatten (Düzleştirme)
  - Fully Connected Layers
  - Dropout (%50 - Overfitting önlemek için)
- **Çıkış:** 3 Sınıf (Softmax)

## 📝 Lisans
Bu proje eğitim amaçlı hazırlanmıştır.
