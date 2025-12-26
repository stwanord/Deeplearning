DERİN ÖĞRENME TABANLI YANGIN VE DUMAN ALGILAMA SİSTEMİ
1. Proje Konusu ve Motivasyonu
Giriş ve Önem: Doğal afetler arasında yer alan orman yangınları ve endüstriyel tesislerde meydana gelen yangın kazaları, erken müdahale edilmediği takdirde ekosistem ve insan yerleşkeleri üzerinde geri döndürülemez tahribatlara yol açmaktadır. Geleneksel duman dedektörleri, dumanın sensöre fiziksel olarak ulaşmasını gerektirdiği için açık alanlarda veya yüksek tavanlı yapılarda tepki süresi açısından yetersiz kalmaktadır. Bu çalışmada, görsel verileri analiz ederek yangını henüz başlangıç safhasında tespit edebilen bir derin öğrenme modeli geliştirilmesi hedeflenmiştir.

Literatür ve Uygulama Gerekçesi: Geçmişte bu alanda yapılan çalışmalar genellikle RGB veya HSV gibi renk uzayları üzerinden yapılan eşikleme yöntemlerine dayanmaktaydı. Ancak bu yöntemler, gün batımı ışığı veya yapay aydınlatmalar gibi durumlarda yüksek oranda "yanlış alarm" üretmekteydi. Günümüzde ise Evrişimli Sinir Ağları (CNN), görsel dokuları ve karmaşık desenleri (alev dili, duman yoğunluğu vb.) tanıma yeteneği ile bu problemin çözümünde altın standart haline gelmiştir. Bu proje, modern görüntü işleme tekniklerinin güvenlik sistemlerine entegrasyonu açısından kritik bir öneme sahiptir.

2. Veri Setinin Yapılandırılması ve Ön İşleme
Çalışma kapsamında, modelin farklı çevresel koşullarda yüksek doğrulukla çalışabilmesi için üç ana sınıftan oluşan bir veri seti kullanılmıştır:

Yangın (Fire): Çeşitli ışık ve arka plan koşullarında alev görüntüleri.

Duman (Smoke): Yangın öncesi veya başlangıcındaki duman emisyonlarını içeren görseller.

Normal (Neutral): Yanlış pozitif tahminleri engellemek adına doğa, şehir ve iç mekan manzaraları.

Ön İşleme Süreçleri: Modelin giriş katmanıyla uyumlu olması için tüm görseller 224x224 piksel boyutuna getirilmiştir. Eğitim sırasında modelin kararlılığını artırmak amacıyla ImageNet ortalamaları baz alınarak normalizasyon işlemi uygulanmıştır. Ayrıca, sınırlı veri setlerinde aşırı öğrenmeyi (overfitting) engellemek ve modelin genelleme yeteneğini artırmak için eğitim aşamasında yatay çevirme (flipping) ve rastgele döndürme (rotation) gibi veri artırma (augmentation) tekniklerinden yararlanılmıştır.

3. Yöntem Seçimi ve Teknik Analiz
Projede ana algoritma olarak Evrişimli Sinir Ağları (CNN) tercih edilmiştir.

Karşılaştırmalı Analiz ve Seçim Nedeni:

CNN vs. Klasik ANN: Klasik yapay sinir ağları, görüntüyü düz bir vektör olarak kabul eder ve pikseller arasındaki mekansal ilişkiyi kaybeder. CNN ise filtreleme mantığıyla görüntüdeki "kenar", "köşe" ve "doku" bilgilerini korur.

Geleneksel Görüntü İşleme vs. Derin Öğrenme: Geleneksel yöntemlerde özellikler (features) elle tanımlanırken, CNN bu özellikleri eğitim sürecinde kendisi optimize eder.

Model Mimarisi: PyTorch kütüphanesi kullanılarak tasarlanan model; üç adet evrişimli blok (Convolution, ReLU, MaxPooling), bir düzleştirme (Flatten) katmanı ve sınıflandırma için tam bağlantılı (Fully Connected) katmanlardan oluşmaktadır. Modelin güvenilirliğini artırmak adına, nöronların yarısını rastgele pasif bırakan %50 Dropout mekanizması entegre edilmiştir.

4. Model Eğitimi ve Değerlendirme
Eğitim süreci, çok sınıflı sınıflandırma problemlerinde verimliliği kanıtlanmış olan CrossEntropyLoss hata fonksiyonu ve Adam optimizer (öğrenme hızı: 0.001) kullanılarak yürütülmüştür.

Bulgular:

Model, 10 epoch süren eğitim sonunda doğrulama (validation) seti üzerinde %85'in üzerinde bir doğruluk (accuracy) oranına ulaşmıştır.

Eğitim kaybı (Loss) grafiği incelendiğinde, modelin hata payının her aşamada istikrarlı bir şekilde azaldığı gözlemlenmiştir.

Modelin, özellikle "Normal" sahneler ile "Duman" arasındaki ince farkları ayırt edebilme becerisi, sistemin güvenilirliğini kanıtlar niteliktedir. Elde edilen başarım grafikleri (training_results.png), modelin hem eğitim hem de test verisi üzerinde dengeli bir performans sergilediğini göstermektedir.
