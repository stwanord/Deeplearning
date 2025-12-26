DERİN ÖĞRENME TABANLI YANGIN VE DUMAN ALGILAMA SİSTEMİ

1. Proje Konusu ve Motivasyonu
   
Seçilme Gerekçesi ve Alanın Önemi: Yangınlar, özellikle ormanlık alanlarda ve endüstriyel tesislerde çok kısa sürede kontrol edilemez boyutlara ulaşarak büyük can ve mal kayıplarına neden olmaktadır. Geleneksel duman dedektörleri, dumanın sensöre fiziksel olarak temas etmesini gerektirdiğinden açık alanlarda veya çok geniş kapalı mekanlarda tepki süresi açısından yetersiz kalmaktadır. Bu projenin temel motivasyonu, kamera görüntüleri üzerinden yapay zeka desteğiyle yangını henüz başlangıç aşamasında (alev veya duman formunda) tespit ederek, erken uyarı sistemlerine otonom ve hızlı bir veri kaynağı sağlamaktır.

Literatürdeki Önceki Çalışmalar: Geçmişte bu alanda yapılan çalışmalar genellikle RGB veya HSV renk uzayları üzerinden yapılan "renk tabanlı eşikleme" yöntemlerine dayanmaktaydı. Ancak bu yöntemler; güneş ışığı, yapay aydınlatmalar veya turuncu/kırmızı nesneleri yangınla karıştırarak yüksek oranda hatalı alarm (false positive) üretmekteydi. Günümüzde ise Evrişimli Sinir Ağları (CNN), görsel dokuları ve karmaşık desenleri tanıma yeteneği ile bu problemin çözümünde en başarılı yaklaşım olarak kabul edilmektedir.

2. Veri Seti Belirlenme Kriterleri
   
Projede kullanılan veri seti, modelin gerçek hayat senaryolarında karşılaşabileceği çeşitliliği yansıtacak şekilde üç ana kategoriye ayrılmıştır:

Yangın (Fire): Farklı yoğunluk, açı ve ışık koşullarındaki alev görüntüleri.

Duman (Smoke): Yangının ilk evrelerinde ortaya çıkan farklı renk ve yoğunluktaki duman emisyonları.

Normal (Neutral): Yanlış alarmları en aza indirmek için seçilen; orman, şehir, bina içi ve gökyüzü gibi yangın içermeyen negatif örnekler.

Veri Hazırlık Süreci: Modelin giriş katmanıyla uyum sağlaması için tüm görseller 224x224 piksel boyutuna getirilmiştir. Eğitim sırasında modelin kararlılığını artırmak amacıyla piksel değerleri normalize edilmiştir. Ayrıca, modelin sadece belirli pozisyonlara ezber yapmasını önlemek ve genelleme yeteneğini artırmak için eğitim aşamasında yatay çevirme (horizontal flip) ve rastgele döndürme (rotation) gibi veri artırma (augmentation) teknikleri uygulanmıştır.

Dataset kaynağı:https://github.com/DeepQuestAI/Fire-Smoke-Dataset

3. Yöntem Analizi ve Algoritma Seçimi
   
Bu çalışmada derin öğrenme mimarilerinden Evrişimli Sinir Ağları (CNN) tercih edilmiştir.

Literatür Karşılaştırması ve Seçim Nedeni:

CNN vs. Geleneksel Görüntü İşleme: Geleneksel yöntemlerde özellikler (features) elle (manual) tanımlanırken, CNN bu özellikleri eğitim sürecinde filtreler aracılığıyla kendi optimize eder. Bu durum, karmaşık yangın sahnelerinde çok daha yüksek doğruluk sağlar.

CNN vs. Klasik Yapay Sinir Ağları (ANN): Klasik sinir ağları görüntüyü düz bir vektör olarak ele alır ve pikseller arasındaki mekansal (spatial) ilişkiyi kaybeder. CNN ise evrişim filtreleri sayesinde görüntüdeki lokal desenleri (alev dokusu, duman dağılımı vb.) korur.

Model Mimarisi: PyTorch framework'ü kullanılarak tasarlanan model; üç adet evrişimli blok (Convolution, ReLU aktivasyonu, MaxPooling), bir düzleştirme (Flatten) katmanı ve sınıflandırma için tam bağlantılı (Fully Connected) katmanlardan oluşmaktadır. Ayrıca aşırı öğrenmeyi (overfitting) engellemek adına %50 oranında Dropout katmanı kullanılmıştır.

4. Model Eğitimi ve Değerlendirmesi
   
Eğitim Parametreleri: Modelin eğitimi, çok sınıflı sınıflandırma problemlerinde verimliliği kanıtlanmış olan CrossEntropyLoss hata fonksiyonu ve Adam optimizer (öğrenme hızı: 0.001) ile yürütülmüştür. Eğitim süreci toplam 10 epoch olarak planlanmış ve her batch'te 32 görüntü işlenmiştir.

5.Performans Analizi:

Doğruluk (Accuracy): Model, eğitim sonunda doğrulama (validation) seti üzerinde %85'in üzerinde bir başarı oranına ulaşmıştır.

Hata Payı (Loss): Eğitim kaybı grafiklerinde görüldüğü üzere, model her epoch sonunda istikrarlı bir şekilde yakınsama (convergence) göstermiştir.

Sonuç: Modelin özellikle duman ile bulut/pus arasındaki farkı ve alev ile yoğun ışık kaynaklarını ayırt edebilme becerisi, sistemin güvenilirliğini kanıtlamaktadır. Elde edilen training_results.png dosyası, eğitim ve test aşamalarındaki dengeli performansı doğrulamaktadır.
Gradyo için: python -u predict.py
