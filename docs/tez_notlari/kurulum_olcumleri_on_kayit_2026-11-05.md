# Ön kayıt: "sahaya kurulabilir sistem" çerçevesi için üç ölçüm (5 Kasım 2026, sonuç görülmeden yazıldı)

## Neden

Makalenin manşeti hız olamaz. Fast-AED (1480 FPS) ve AED-MAE (1655 FPS) hem çok daha hızlı hem de Ped2 ve Avenue'da daha doğru. Bu yüzden öne çıkarmak istediğimiz şey şu: sistem sahaya kurulabilir. Ağ eğitimi yok, etiket yok, her veri setinde tek ayar var, sistem canlı çalışıyor ve alarm eşiği etiket görmeden seçiliyor.

Bu iddia ancak ölçülürse savunulabilir. Aşağıdaki üç ölçüm bunun için. Kurallar:
- Sonuçlar olumsuz çıksa da raporlanır.
- Test etiketleri hiçbir seçimde kullanılmaz.

**Sistem:** Tur 2'nin `base` yapılandırması (`vadkit/configs/r2/base.yaml`; ShT için `base_sht.yaml`). Bileşenler YOLO11n, YOLO11n-pose, DIS ve MobileCLIP2-S0. Puanlayıcı NIW, kalibrasyon Fisher, füzyon toplam, ardından nedensel medyan 9; her videonun ilk 4 karesi atılır. Yapılandırma seçimi yapılmaz.

## Karşılaştırma tablolarının düzeni (hocanın kuralı)

Her karşılaştırma elma ile elma yapılır. Yöntemler denetim türüne göre ayrı bloklarda verilir:
1. Yalnız normal veriyle eğitilen (tek sınıf)
2. Eğitimsiz, yani hazır modellerle çalışan
3. Sentetik ya da sözde anomaliyle eğitilen (ör. AED-MAE)
4. Zayıf denetimli (video düzeyinde etiket)

Protokol ayrıca ayrılır: resmî (çevrimdışı) ve canlı (nedensel).

Bizim sistem nesne dalıyla blok 2'ye, normal veriyle eğitilen ızgara AE'si eklendiğinde blok 1'e girer.

## M1. Etiketsiz alarm eşiği (yanlış alarm kalibrasyonu)

**Eşik nasıl seçilir:**
1. Nesne dalının çapraz uydurulmuş normal referansı alınır. Bu referans eğitim videolarının 2 katlı bölünmesiyle elde edilir, her kare için her kanalda vardır.
2. Her kanal Fisher ile kalibre edilir, kanallar toplanır.
3. Video başına önce ilk 4 kare atılır, sonra nedensel medyan 9 uygulanır. Bu, test skorlarıyla aynı işlemdir.
4. Eşik τ_α, bu dağılımın (1 − α) kantilidir. α ∈ {0.1, 0.05, 0.01, 0.001}.

Eşik her kamera için ayrı seçilir: Ped1, Ped2 ve Avenue'nun her biri bir kamera, ShT'de ise her sahne bir kamera.

**Ölçülenler:**
- Test videolarındaki normal karelerde (gt = 0) gerçekleşen yanlış alarm oranı: FAR = P(s > τ_α).
- Aynı eşikte anomali karelerindeki (gt = 1) tespit oranı (TPR).
- Test videoları üzerinde 2000 tekrarlı bootstrap ile %95 güven aralığı.

**Ana küme:** Ped1, Ped2 ve Avenue. Bu setlerde eğitim referansı tam kare hızında.

**ShT (ikincil, işaretli):** Eğitim referansı her 5. karede alınıyor. Bu yüzden medyan, alt örneklenmiş sıra üzerinde uygulanır ve bu bir yaklaşıklıktır.

**Ek karşılaştırmalar:**
- Kanalların bağımsız olduğu varsayımıyla analitik eşik: toplam ~ Gamma(k, 1). Nesne dalında k = 3.
- Izgara + nesne (k = 6): Izgara referansları kare hizalı değil (dosya başına 154–850 örnek), bu yüzden bu sistem için yalnız analitik eşik raporlanır. İkincil sonuçtur.

**Önceden yazılan ölçüt:** α = 0.05 ve α = 0.01 için FAR / α oranı 0.5 ile 2 arasındaysa eşik "kalibre" sayılır. Hipotez, bunun ana kümenin en az 2/3'ünde tutmasıdır.

## M2. Kurulum maliyeti ve veri verimliliği

**Kurulum süresi:** Üç kalemden oluşur:
1. Normal eğitim videolarında özellik çıkarımı. Tur 2'deki kare başı gecikme × eğitim kare sayısı.
2. Uydurma süresi. CPU duvar saati, iş parçacığı sayısı kaydedilir. Tur 2 bittikten sonra sessiz makinede ölçülür.
3. Ağ eğitimi. Nesne dalında yoktur. Izgara dalının AE eğitim süresi arşiv günlüklerinde varsa raporlanır, yoksa "ölçülmedi" yazılır.

**Veri verimliliği:**
- Eğitim videolarından f ∈ {0.1, 0.25, 0.5, 1.0} oranında alt küme alınır.
- Video sayısı n = max(2, ⌈f · N⌉). En az 2 video gerekir, çünkü seçim 2 katlıdır. ShT'de bu kural sahne başına uygulanır.
- Her oran için 5 tohum kullanılır: videolar `default_rng(1000 + tohum)` ile seçilir. f = 1.0 tek koşudur.
- Yapılandırma aynıdır, test setinin tamamı kullanılır.
- Raporlanan: nedensel mikro ve makro AUC (ortalama ± std) ve kullanılan normal videonun dakika cinsinden süresi. Kare hızları: UCSD 10 FPS, Avenue 25 FPS, ShT 24 FPS.

**Hipotez:** f = 0.25 ile mikro AUC, tam verinin en fazla 2 puan altındadır. Bunun Ped1, Ped2, Avenue ve ShT'nin en az 3/4'ünde tutması beklenir.

## M3. Rakiplerin canlı protokolde yeniden değerlendirilmesi

**Rakipler:**
- Fast-AED ve AED-MAE: hız iddiasının doğrudan rakipleri.
- VideoPatchCore: eğitimsiz sistem iddiasının doğrudan rakibi.

Üçünün de kodu açık.

**Adımlar:**
1. **Sadakat kontrolü.** Reponun kendi son işlemesiyle makaledeki sayıya ±1 puan içinde yaklaşılmalı. Yaklaşılamazsa sonuç "yeniden üretilemedi" diye raporlanır, sayısıyla birlikte.
2. **Canlı protokol.** Kare başına ham skor kullanılır. Video başına normalizasyon ve merkezli Gauss ya da 3B filtre çıkarılır. Normalizasyon yalnız normal eğitim verisinin istatistiğiyle yapılır; eğitim skoru üretilemiyorsa normalizasyon yapılmaz. Ardından nedensel medyan 9 uygulanır ve her videonun ilk 4 karesi atılır. Mikro ve makro AUC raporlanır.
3. **İleri bakış.** Yöntem t karesinin skorunda t'den sonraki kareleri kullanıyorsa bu düzeltilemez. Tabloda gecikme olarak raporlanır.
4. **Veri setine göre ayar.** Yöntemin kendi ayarları olduğu gibi bırakılır ve tabloda işaretlenir.
5. **Hız.** Uçtan uca FPS aynı RTX 3090'da ölçülür; ön işleme dahil edilir.

**Hipotez:** Video başına normalizasyon kullanan yöntemler, canlı protokolde Avenue'da resmî sayılarından en az 2 puan kaybeder.
