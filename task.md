# Derin Öğrenme Dönem Projesi Görev Listesi: Yangın Tespiti

## 1. Proje Konusu (15 Puan)
- [x] Proje konusunun seçilmesi: **Yangın ve Duman Tespiti (Fire & Smoke Detection)**
- [ ] Seçilme gerekçesinin yazılması (Orman yangınları, erken uyarı sistemleri, can/mal güvenliği)
- [ ] Literatür taraması (Yangın tespitinde kullanılan CNN yöntemleri)
- [ ] Alanın önemi (Akıllı şehirler, IoT entegrasyonu)

## 2. Veri Setinin Belirlenmesi (15 Puan)
- [x] Veri setinin indirilmesi (Kaggle Fire/Non-Fire Dataset veya benzeri)
- [x] Veri setinin düzenlenmesi (Train/Test klasörlerine ayrılması)
- [x] Veri ön işleme (Resize, Normalization, Augmentation - Döndürme/Kırpma ile veri çoğaltma)

## 3. Yöntem/Algoritma Seçimi (15 Puan)
- [ ] Yöntem: **CNN (Convolutional Neural Networks)**
- [ ] Mimari: Sınıflandırma başarısını artırmak için Dropout ve Batch Normalization katmanları eklenecek.
- [ ] Karşılaştırma: Basit CNN vs. Transfer Learning (VGG16 veya ResNet) karşılaştırması (Opsiyonel ama puan getirir).

## 4. Model Eğitimi & Değerlendirme (20 Puan)
- [x] Veri Yükleyici (DataLoader) kodlaması
- [x] CNN Modelinin PyTorch ile oluşturulması
- [x] Modelin eğitilmesi (Training Loop)
- [x] Başarı değerlendirmesi (Accuracy, Loss grafikleri)
- [/] Karışıklık Matrisi (Confusion Matrix) çizimi - **Rapora eklenecek**

## 5. Proje Dokümantasyonu (10 Puan)
- [x] Kodların modüler hale getirilmesi (`dataset.py`, `model.py`, `train.py`)
- [x] `readme.md` dosyasının projenin amacını, veriyi ve nasıl çalıştırılacağını anlatacak şekilde yazılması
- [x] Proje raponun yazılması

## 6. Sunum (25 Puan)
- [x] Sunum materyallerinin (görseller, grafikler) hazırlanması
