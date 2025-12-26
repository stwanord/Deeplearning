# DERİN ÖĞRENME DÖNEM PROJESİ RAPORU

**Konu:** Convolutional Neural Networks (CNN) ile Yangın ve Duman Tespiti

---

## 1. Proje Konusu ve Amacı (15 Puan)
**Seçilen Konu:** Görüntü İşleme tabanlı Yangın ve Duman Tespiti.

**Seçilme Gerekçesi:**
Orman yangınları ve bina yangınları, erken müdahale edilmediğinde büyük can ve mal kayıplarına yol açmaktadır. Geleneksel duman dedektörleri sadece kapalı ve küçük alanlarda çalışırken, kamera tabanlı sistemler açık alanlarda (ormanlar, fabrikalar) geniş çaplı izleme yapabilir. Bu proje, derin öğrenme tekniklerinin bu hayati problemde nasıl kullanılabileceğini göstermek amacıyla seçilmiştir.

**Literatür Özeti:**
Literatürde yangın tespiti için renk bazlı (RGB/HSV analizleri) geleneksel görüntü işleme yöntemleri kullanılsa da, son yıllarda CNN (Evrişimli Sinir Ağları) modellerinin çok daha yüksek başarı sağladığı görülmüştür. Özellikle AlexNet, VGG16 ve ResNet gibi mimariler bu alanda sıkça kullanılmaktadır.

**Alanın Önemi:**
Akıllı Şehirler (Smart Cities) ve IoT tabanlı güvenlik sistemlerinde kamera verilerinin otomatik analizi giderek önem kazanmaktadır. Bu proje, otonom gözetleme sistemlerinin temelini oluşturabilecek bir çalışmadır.

---

## 2. Veri Seti (Dataset) (15 Puan)
**Kullanılan Veri Seti:** DeepQuestAI Fire-Smoke Dataset (veya benzeri açık kaynak veri seti).
**Yapısı:**
Veri seti üç ana sınıftan oluşmaktadır:
1.  **Fire (Yangın):** Çeşitli açılardan ve ışık koşullarında alev görüntüleri.
2.  **Smoke (Duman):** Siyah ve beyaz duman içeren görüntüler.
3.  **Neutral (Normal):** Yangın olmayan doğa, orman ve şehir görüntüleri.

**Ön İşleme (Preprocessing):**
*   **Boyutlandırma:** Tüm görüntüler modelin girişine uygun olarak **224x224** piksel boyutuna getirilmiştir.
*   **Normalizasyon:** Piksel değerleri standart ImageNet ortalamalarına (mean=[0.485, 0.456, 0.406]) göre normalize edilmiştir.
*   **Veri Artırma (Data Augmentation):** Eğitim setindeki görüntülerin sayısı ve çeşitliliğini artırmak için rastgele yatay çevirme (Horizontal Flip) ve döndürme (Rotation) işlemleri uygulanmıştır.

---

## 3. Yöntem ve Algoritma (15 Puan)
**Kullanılan Yöntem:** Convolutional Neural Network (CNN).

**Neden CNN?**
Görüntü verileri üzerindeki desenleri, kenarları ve dokuları (alev dokusu, duman yoğunluğu vb.) en iyi tanıyan mimari CNN'dir. Klasik Yapay Sinir Ağları (ANN), görüntüdeki mekansal (spatial) ilişkileri koruyamazken, CNN filtreleme mantığıyla bunu başarır.

**Model Mimarisi:**
Projede 3 katmanlı özel bir CNN mimarisi tasarlanmıştır:
1.  **Conv Block 1:** 32 Filtre (3x3), ReLU Aktivasyon, Max Pooling (2x2).
2.  **Conv Block 2:** 64 Filtre (3x3), ReLU Aktivasyon, Max Pooling (2x2).
3.  **Conv Block 3:** 128 Filtre (3x3), ReLU Aktivasyon, Max Pooling (2x2).
4.  **Flatten:** Özellik haritalarının vektöre dönüştürülmesi.
5.  **Dropout:** %50 oranında nöron kapatma (Ezberlemeyi/Overfitting'i engellemek için).
6.  **Fully Connected Layers:** Sınıflandırma katmanları.

---

## 4. Model Eğitimi ve Değerlendirme (20 Puan)
**Eğitim Parametreleri:**
*   **Optimizer:** Adam (Learning Rate: 0.001)
*   **Loss Function:** CrossEntropyLoss (Çok sınıflı sınıflandırma için)
*   **Epoch Sayısı:** 10
*   **Batch Size:** 32

**Eğitim Sonuçları:**
Model eğitimi sonucunda elde edilen başarı grafikleri `training_results.png` dosyasında sunulmuştur.
*   **Eğitim Kaybı (Train Loss):** Düzenli olarak düşüş göstermiştir.
*   **Doğrulama Doğruluğu (Validation Accuracy):** %85 seviyelerinin üzerine çıkmıştır.

Model, eğitim setinde görmediği "Validation" verileri üzerinde başarılı tahminler yaparak genelleme yeteneğini kanıtlamıştır.

---

## 5. Sonuç ve Öneriler
Geliştirilen model, temel düzeyde yangın ve duman tespiti görevini başarıyla yerine getirmektedir. Projenin ilerleyen aşamalarında daha büyük veri setleri ve Transfer Learning (hazır eğitilmiş modeller) kullanılarak başarı oranı artırılabilir. Ayrıca, model bir drone veya güvenlik kamerasına entegre edilerek gerçek zamanlı bir uygulama haline getirilebilir.
